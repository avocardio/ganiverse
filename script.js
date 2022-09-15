// variable declaration

/*

TensorFlow.js Layers model: JSON + binary weight file(s), with limited (Keras) features. 
The weights seem to be optional in this case. And from the Tensorflow JS docs,

- This mode is not applicable to TensorFlow SavedModels or their converted forms. 
For those models, use tf.loadGraphModel(). The loaded model supports the full inference 
and training (e.g., transfer learning) features of the original keras or tf.keras model.

TensorFlow.js graph model: JSON + binary weight file(s), with conversion to/ from SavedModel, 
but no training capability. About the graph model, the README says:

The loaded model supports only inference, but the speed of inference is generally faster than 
that of a tfjs_layers_model (see above row) thanks to the graph optimization performed by 
TensorFlow. Another limitation of this conversion route is that it does not support some layer 
types (e.g., recurrent layers such as LSTM) yet.

NotImplementedError: Saving the model to HDF5 format requires the model to be a Functional 
model or a Sequential model. It does not work for subclassed models, because such models are 
defined via the body of a Python method, which isn't safely serializable. Consider saving to 
the Tensorflow SavedModel format (by setting save_format="tf") or using `save_weights`.

! When converting, supported quantization dtypes are: 'uint8' and 'uint16'.

___________ 


I could avoid the error with the model by following these steps:

1. saving the tensorflow model with .h5 format:

model.save('sampleModel.h5')

2. converting the .h5 file for use in tensorflow.js as tfjs_graph_model:

tensorflowjs_converter --input_format keras --output_format tfjs_graph_model 
sampleModel.h5 sampleModel

3. using loadGraphModel() instead of loadLayerModel() for loading the model

*/

// tf.enableDebugMode()

var canvas = document.getElementById("canvas");
var cW = canvas.width;
var cH = canvas.height;
var ctx = canvas.getContext('2d');
var dragging = false;
var pos = { x: 0, y: 0 };
var slider = document.getElementById("slider");
var output = document.getElementById("output");
document.getElementById('model_checker').innerHTML="Please wait for the model to load. (This can take up to 20 seconds)"
output.innerHTML = "Mean of Z: " + slider.value / 100;

slider.oninput = function() {
    output.innerHTML = "Mean of Z: " + this.value /100;
}

function erase() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function opencvcheck() {
    document.getElementById('checker').innerHTML="Opencv is ready."
}

async function loadModel(){	
  	
    // loads the model
    generator = await tf.loadGraphModel('model/model.json');    

    const input = tf.randomNormal([1, 100]);
    
    // warm start the model. speeds up the first inference
    await generator.predict(input);
    
    document.getElementById('model_checker').innerHTML="Model is ready."
}

async function generateCustom() {

    await ctx.clearRect(0, 0, canvas.width, canvas.height);

    let input = document.getElementById("slider").value
    console.log(`Random Noisy Input Mean is ${input/100}`)

    inputtensor = tf.randomNormal([1, 100], input/100, 0.2);
    console.log(`Predicting: ${inputtensor}`)

    // Prediction
    outputtensor = await generator.predict(inputtensor)

    outputtensor = outputtensor.reshape([500, 500, 3])
    outputtensor = tf.concat([outputtensor, tf.ones([500, 500, 1])], 2)
    result = outputtensor.dataSync()

    //result = outputtensor.mul([1, 1, 1]).dataSync()
    for(var i=0;i<result.length;i++){
        result[i]=result[i]*255.0 + 128.0;
    }

    ctx.putImageData(new ImageData(Uint8ClampedArray.from(result), 500, 500), 0, 0);
}

async function generateRandom() {

    await ctx.clearRect(0, 0, canvas.width, canvas.height);

    inputtensor = tf.randomNormal([1, 100]);
    console.log(`Predicting: ${inputtensor}`)

    // Change slider value to the mean of the input tensor
    document.getElementById("slider").value = inputtensor.mean().dataSync()[0] * 100;

    document.getElementById('output').innerHTML = "Mean of Z: " + Math.round(inputtensor.mean().dataSync()[0] * 100) / 100;

    // Prediction
    outputtensor = await generator.predict(inputtensor)

    outputtensor = outputtensor.reshape([500, 500, 3])
    outputtensor = tf.concat([outputtensor, tf.ones([500, 500, 1])], 2)
    result = outputtensor.dataSync()

    //result = outputtensor.mul([1, 1, 1]).dataSync()
    for(var i=0;i<result.length;i++){
        result[i]=result[i]*255.0 + 128.0;
    }

    ctx.putImageData(new ImageData(Uint8ClampedArray.from(result), 500, 500), 0, 0);
}

var generator;

tf.loadGraphModel('model/model.json').then(async (resolve) => {
    generator=resolve
});

// Function for applying OpenCV filters
async function applyFilter(tensor) {
    let input = tensor.reshape([500, 500, 3])
    input = tf.concat([input, tf.ones([500, 500, 1])], 2)
    input = input.dataSync()
    let filter = cv.bilateralFilter(input, 5, 50, 50, cv.BORDER_DEFAULT)
    return filter
}