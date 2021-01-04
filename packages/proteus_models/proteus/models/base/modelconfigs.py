import pydantic


class BaseModelConfig(pydantic.BaseModel):
    num_instances: int = 1


class TritonOptimizationModelConfig(pydantic.BaseModel):
    triton_optimization: bool = True


class QuantizationModelConfig(pydantic.BaseModel):
    quantize: bool = False


class BatchingModelConfig(pydantic.BaseModel):
    dynamic_batching: bool = True
