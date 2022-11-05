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

var tfmodel = 'model/model.json';

function isMobile() {
    return /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent);
}

if (isMobile()) {
   tfmodel = 'model2/model.json';
}

function showPlanet() {
    const $planet = document.querySelector('.planet')
    // Animate the planet on load
    $planet?.classList.remove('slide-in')
    setTimeout(() => {
        $planet.classList.add('slide-in')
        $planet.style.display = 'block'
    })
}

function showScanner() {
    const $scanner = document.querySelector('.scanner')
    $scanner.style.display = 'block'
}

function hideScanner() {
    const $scanner = document.querySelector('.scanner')
    $scanner.style.display = 'none'
}

var canvas = document.getElementById("canvas");
var cW = canvas.width;
var cH = canvas.height;
var ctx = canvas.getContext('2d');
var dragging = false;
var pos = { x: 0, y: 0 };
var slider = document.getElementById("slider");
var output = document.getElementById("output");
var name = document.getElementById("name");
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

function name_planet() {
    document.getElementById('name').innerHTML= nameGenerator();
}

async function loadModel(){	

    // loads the model
    generator = await tf.loadGraphModel(tfmodel);  

    const input = tf.randomNormal([1, 100]);
    
    // warm start the model. speeds up the first inference
    await generator.predict(input);
    
    document.getElementById('model_checker').innerHTML="Model is ready."
}

async function generateCustom() {

    showScanner();

    if (tfmodel == 'model/model.json') {
        var dims = 500;
        document.getElementById("canvas").width = dims;
        document.getElementById("canvas").height = dims;
    }
    else if (tfmodel == 'model2/model.json') {
        var dims = 256;
        document.getElementById("canvas").width = dims;
        document.getElementById("canvas").height = dims;
    }

    await ctx.clearRect(0, 0, canvas.width, canvas.height);

    let input = document.getElementById("slider").value
    console.log(`Random Noisy Input Mean is ${input/100}`)

    inputtensor = tf.randomNormal([1, 100], input/100, 0.2);
    console.log(`Predicting: ${inputtensor}`)

    // Prediction
    outputtensor = await generator.predict(inputtensor)

    outputtensor = outputtensor.reshape([dims, dims, 3])
    outputtensor = tf.concat([outputtensor, tf.ones([dims, dims, 1])], 2)
    result = outputtensor.dataSync()

    //result = outputtensor.mul([1, 1, 1]).dataSync()
    for(var i=0;i<result.length;i++){
        result[i]=result[i]*255.0 + 128.0;
    }

    ctx.putImageData(new ImageData(Uint8ClampedArray.from(result), dims, dims), 0, 0);

    name_planet();
    document.getElementById("name").style.display = "block";
    hideScanner();
}

async function generateRandom() {

    showScanner();
    
    if (tfmodel == 'model/model.json') {
        var dims = 500;
        document.getElementById("canvas").width = dims;
        document.getElementById("canvas").height = dims;
    }
    else if (tfmodel == 'model2/model.json') {
        var dims = 256;
        document.getElementById("canvas").width = dims;
        document.getElementById("canvas").height = dims;
    }

    await ctx.clearRect(0, 0, canvas.width, canvas.height);

    inputtensor = tf.randomNormal([1, 100]);
    console.log(`Predicting: ${inputtensor}`)

    // Change slider value to the mean of the input tensor
    document.getElementById("slider").value = inputtensor.mean().dataSync()[0] * 100;
    document.getElementById('output').innerHTML = "Mean of Z: " + Math.round(inputtensor.mean().dataSync()[0] * 100) / 100;

    // Prediction
    outputtensor = await generator.predict(inputtensor)

    outputtensor = outputtensor.reshape([dims, dims, 3])
    outputtensor = tf.concat([outputtensor, tf.ones([dims, dims, 1])], 2)
    result = outputtensor.dataSync()

    //result = outputtensor.mul([1, 1, 1]).dataSync()
    for(var i=0;i<result.length;i++){
        result[i]=result[i]*255.0 + 128.0;
    }

    // Apply OpenCV filter
    // result = await applyFilter(result)

    ctx.putImageData(new ImageData(Uint8ClampedArray.from(result), dims, dims), 0, 0);

    name_planet();
    document.getElementById("name").style.display = "block";
    hideScanner();
}

// Function for applying the OpenCV filter cv.bilateralFilter(result, 5, 50, 50, cv.BORDER_DEFAULT) to the generated image
async function applyFilter() {
    let img = cv.imread('canvas');
    cv.cvtColor(img, img, cv.COLOR_RGBA2RGB, 0);
    cv.bilateralFilter(img, img, 5, 50, 50, cv.BORDER_DEFAULT);
    cv.imshow('canvas', img);
    img.delete();
}

// Function that produces a mix of the letters 'gan' and the word 'Tetraodontidae'
function nameGenerator() {
    const parent1 = 'tetradontidae';
    const parent2 = 'gan';
    const n = Math.floor(Math.random() * 3) + 2;
    const chunks = [];
    for (let i = 0; i < parent1.length; i += n) {
        chunks.push(parent1.slice(i, i + n));
    }
    const sampleList = [];
    for (let i = 0; i < 2; i++) {
        const randomIndex = Math.floor(Math.random() * chunks.length);
        sampleList.push(chunks[randomIndex]);
    }
    sampleList.push(parent2);
    sampleList.sort(() => Math.random() - 0.5);
    const name = sampleList.join('');
    const num = Math.floor(Math.random() * 84);

    return name + '-' + num;
}

// Function for saving the image with the generated name in the getelementbyid('name') element
function saveImage() {
    var name = document.getElementById("name");
    var link = document.createElement('a');
    link.download = name.innerHTML + '.png';
    link.href = document.getElementById('canvas').toDataURL()
    link.click();
}