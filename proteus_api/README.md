# apiclient
FastAPI for Triton

Upon start-up, model discovery will happen by looking through the proteus.models namespace and registering all available models.  
For each model a load, unload and predict endpoint will be generated.

Models can be loaded with certain configs, if exposed by the model.  
- num_instances: Triton allows to put multiple copies of the same model on a GPU
- triton_optimization: Whether to use Triton Optimization.  This is recommended, but older Onnx models may not support it.
- quantize: You can quantize models to make them smaller.  There seems to be no performance benefit using Triton, but they take less space.
- dynamic batching: If the model allows batching, Triton can batch dynamically if too many concurrent requests come in.