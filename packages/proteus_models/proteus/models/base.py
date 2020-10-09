import logging
import tritonclient.http as httpclient

# TODO add details on module/def in logger?
logger = logging.getLogger("gunicorn.error")

class BaseModel:

    # Defaults
    MODEL_NAME = 'base'
    MODEL_VERSION = '1'
    MAX_BATCH_SIZE = 1
    NUM_OUTPUTS = 1

    @classmethod
    def parse_model_http(cls, model_metadata, model_config):
        """
        TODO check if this is still relevant
        Check the configuration of a model to make sure it meets the
        requirements for an image classification network (as expected by
        this client)
        """
        if len(model_metadata['inputs']) != 1:
            raise Exception("expecting 1 input, got {}".format(
                len(model_metadata['inputs'])))
        if len(model_metadata['outputs']) != cls.NUM_OUTPUTS:
            raise Exception("expecting {} outputs, got {}".format(
                cls.NUM_OUTPUTS,
                len(model_metadata['outputs'])))

        if len(model_config['input']) != 1:
            raise Exception(
                "expecting 1 input in model configuration, got {}".format(
                    len(model_config['input'])))

        input_metadata = model_metadata['inputs'][0]
        output_metadatas = model_metadata['outputs']

        for output_metadata in output_metadatas:
            if output_metadata['datatype'] != "FP32":
                raise Exception("expecting output datatype to be FP32, model '" +
                                model_metadata['name'] + "' output type is " +
                                output_metadata['datatype'])

        # sort to make sure that order of outputs is fixed
        output_metadatas = sorted([output_metadata['name']
                                  for output_metadata in output_metadatas])

        # Model input must have 3 dims (not counting the batch dimension),
        # either CHW or HWC
        input_batch_dim = (cls.MAX_BATCH_SIZE > 0)
        expected_input_dims = 3 + (1 if input_batch_dim else 0)
        if len(input_metadata['shape']) != expected_input_dims:
            raise Exception(
                "expecting input to have {} dimensions, model '{}' input has {}".
                format(expected_input_dims, model_metadata['name'],
                       len(input_metadata['shape'])))

        return (input_metadata['name'], output_metadatas, input_metadata['datatype'])

    @classmethod
    def requestGenerator(cls, batched_image_data, input_name, 
                         output_names, dtype):
        """ Set the input data """
        inputs = [httpclient.InferInput(input_name, batched_image_data.shape,
                                        dtype)]
        inputs[0].set_data_from_numpy(batched_image_data, binary_data=True)

        outputs = [httpclient.InferRequestedOutput(output_name,
                                                   binary_data=True)
                   for output_name in output_names]
        yield inputs, outputs