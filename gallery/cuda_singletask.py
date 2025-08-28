import os
# os.environ["USE_DAG_MOD"] = "1"
import numpy as np
from tvm import auto_scheduler
import tvm
from tvm import relay, tir
from tvm.auto_scheduler import SearchTask
from tvm.auto_scheduler import SketchPolicy
from tvm.relay import testing
from tvm.relay.testing.init import create_workload
from tvm.contrib import graph_executor
from tvm.relay.build_module import bind_params_by_name
from tvm.relay import transform

# os.environ["TVM_LOG_DEBUG"] = "relay/backend/te_compiler.cc=1"
# os.environ["TVM_LOG_DEBUG"] = "DEFAULT=0,relay/backend/te_compiler.cc=1"
# os.environ["GLOG_v"] = "1"
# export TVM_LOG_DEBUG="DEFAULT=0"



def get_network(name, batch_size, layout="NHWC", dtype="float32", use_sparse=False):
    """Get the symbol definition and random weight of a network"""

    # auto-scheduler prefers NHWC layout
    if layout == "NHWC":
        image_shape = (224, 224, 3)
    elif layout == "NCHW":
        image_shape = (3, 224, 224)
    else:
        raise ValueError("Invalid layout: " + layout)

    input_shape = (batch_size,) + image_shape
    output_shape = (batch_size, 1000)

    if name.startswith("resnet-"):
        n_layer = int(name.split("-")[1])
        mod, params = testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name.startswith("resnet3d-"):
        n_layer = int(name.split("-")[1])
        mod, params = testing.resnet.get_workload(
            num_layers=n_layer,
            batch_size=batch_size,
            layout=layout,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "mobilenet":
        mod, params = testing.mobilenet.get_workload(
            batch_size=batch_size, layout=layout, dtype=dtype, image_shape=image_shape
        )
    elif name == "squeezenet_v1.1":
        assert layout == "NCHW", "squeezenet_v1.1 only supports NCHW layout"
        mod, params = testing.squeezenet.get_workload(
            version="1.1",
            batch_size=batch_size,
            dtype=dtype,
            image_shape=image_shape,
        )
    elif name == "inception_v3":
        input_shape = (batch_size, 3, 299, 299) if layout == "NCHW" else (batch_size, 299, 299, 3)
        mod, params = testing.inception_v3.get_workload(batch_size=batch_size, dtype=dtype)
    elif name == "mxnet":
        # an example for mxnet model
        from mxnet.gluon.model_zoo.vision import get_model

        assert layout == "NCHW"

        block = get_model("resnet50_v1", pretrained=True)
        mod, params = relay.frontend.from_mxnet(block, shape={"data": input_shape}, dtype=dtype)
        net = mod["main"]
        net = relay.Function(
            net.params, relay.nn.softmax(net.body), None, net.type_params, net.attrs
        )
        mod = tvm.IRModule.from_expr(net)
    elif name == "mlp":
        mod, params = testing.mlp.get_workload(
            batch_size=batch_size, dtype=dtype, image_shape=image_shape, num_classes=1000
        )
    else:
        raise ValueError("Network not found.")

    if use_sparse:
        from tvm.topi.sparse.utils import convert_model_dense_to_sparse

        mod, params = convert_model_dense_to_sparse(mod, params, bs_r=4, random_params=True)

    return mod, params, input_shape, output_shape



def get_mod(network_name, batch_size, layout, single=False, dtype="float32"):

    print("Getting network : ", network_name)

    if network_name.startswith("tiny_convnet"):
        layer_n = int(network_name.split("_")[-1])
        net = relay.testing.resnet.tiny_convnet_nhwc(layer_n=layer_n)        # input shape까지 relay에 이미 정의됨
        mod, params = create_workload(net, single)
        input_shape = (batch_size, 32, 32, 3)
        output_shape = (batch_size, 10)

    elif network_name.startswith("tiny_resnet"):
        net = relay.testing.resnet.tiny_resnet_nhwc()
        mod, params = create_workload(net, single)
        input_shape = (batch_size, 32, 32, 3)
        output_shape = (batch_size, 10)

    else:
        mod, params, input_shape, output_shape = get_network(
            network_name,
            batch_size,
            layout,
            dtype=dtype,
        )


    def basic_convert(mod, params):
        mod = relay.transform.SimplifyInference()(mod)
        mod = relay.transform.InferType()(mod)
        return mod, params

    def experiment(mod, params):
        # cuda = tvm.target.Target("cuda")
        # llvm = tvm.target.Target("llvm")
        # get_cfg = tvm.get_global_func("relay.backend.GetCompilationConfig", allow_missing=True)
        # if get_cfg is not None:
        #     # 보통 (mod, targets or multi-target) 형태를 받는다
        #     cfg = get_cfg(mod, [cuda, llvm])
        #     mod = relay.transform.PlanDevices(cfg)(mod)
        
        # mod = tvm.IRModule.from_expr(mod)
        
        passes = [
            relay.transform.SimplifyInference(),
            relay.transform.FoldConstant(),
            relay.transform.InferType(),
            relay.transform.Inline(),
            relay.transform.InferType(),
            relay.transform.LabelOps(),
            relay.transform.InferType(),
        ]
        with tvm.transform.PassContext(opt_level=3, 
                                    config={"relay.backend.use_auto_scheduler": True,},
                                    disabled_pass={"AutoSchedulerLayoutRewrite"}):
                                    #    ):
            mod = tvm.transform.Sequential(passes)(mod)
            f = mod["inner"].with_attr("Primitive", tir.IntImm("int32", 1))
            mod.update_func(mod.get_global_var("inner"), f)
            mod = transform.InferType()(mod)
        return mod, params


    mod, params = experiment(mod, params)    
    # mod, params = basic_convert(mod, params)

    return mod, params, input_shape, output_shape

single = True


batch = 1
layout = "NHWC"
dtype = "float32"

# tiny_convnet_1, tiny_resnet, resnet-18, resnet-50
model_name = "tiny_convnet_1"
log_file = f"logs/{model_name}_b{batch}_single.json"


mod, params, input_shape, output_shape = get_mod(model_name, batch, layout, single)
print(mod)
target = tvm.target.Target("cuda")

# breakpoint()


print("Lowering..")
from tvm.relay.backend import te_compiler

tecompiler = te_compiler.get()
with tvm.transform.PassContext(
    opt_level=3,
    config={
        "relay.backend.use_auto_scheduler": True,
    },
    disabled_pass={"AutoSchedulerLayoutRewrite"},
):
    # vmcompiler = relay.vm.VMCompiler()
    # vmcompiler.set_params(params)
    # vmcompiler.lower(mod, target)

    cached_func = tecompiler.lower(mod["inner"], target)
print("Lowered")


breakpoint()



print("ComputeDAG..")
# dag = auto_scheduler.ComputeDAG(list(cached_func.outputs))
dag = auto_scheduler.ComputeDAG(list(cached_func.inputs) + list(cached_func.outputs))
print("ComputeDAG done")

key = dag.workload_key()
auto_scheduler.workload_registry.register_workload_tensors(key, dag.tensors)


# breakpoint()
# tune_option = auto_scheduler.TuningOptions(
#     num_measure_trials=1000,  # change this to 20000 to achieve the best performance
#     runner=auto_scheduler.LocalRunner(repeat=10, enable_cpu_cache_flush=True)
# )
tasks = [SearchTask(
    workload_key = key,
    compute_dag=dag,
    target=target
)]

# searchpolicy = SketchPolicy(
#         tasks[0],
#         auto_scheduler.XGBModel(),
#         params=params,
#         verbose=1,
# )

# pop = searchpolicy.sample_initial_population()

breakpoint()

def run_tuning():
    print("Begin tuning...")
    measure_ctx = auto_scheduler.LocalRPCMeasureContext(repeat=1, min_repeat_ms=300, timeout=10)

    tuner = auto_scheduler.TaskScheduler(tasks, [1])
    tune_option = auto_scheduler.TuningOptions(
        num_measure_trials=2000,  # change this to 20000 to achieve the best performance
        runner=measure_ctx.runner,
        measure_callbacks=[auto_scheduler.RecordToFile(log_file)],
    )

    tuner.tune(tune_option)

# run_tuning()



print("Compile...")
with auto_scheduler.ApplyHistoryBest(log_file):
    with tvm.transform.PassContext(opt_level=3, config={"relay.backend.use_auto_scheduler": True}):
        # breakpoint()
        lib = relay.build(mod, target=target, params=params)

print("Compile done")
# Create graph executor
dev = tvm.device(str(target), 0)
breakpoint()
module = graph_executor.GraphModule(lib["default"](dev))
data_tvm = tvm.nd.array((np.random.uniform(size=input_shape)).astype("float32"))
module.set_input("data", data_tvm)

# Evaluate
print("Evaluate inference time cost...")
print(module.benchmark(dev, repeat=3, min_repeat_ms=500))