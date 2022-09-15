// variable declaration

var canvas = document.getElementById("canvas");
var ctx = canvas.getContext('2d');
var dragging = false;
var pos = { x: 0, y: 0 };

function erase() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
}

async function loadModel(){	
  	
    // loads the model
    model = await tf.loadModel('../tfjs/model.json');    
    
    // warm start the model. speeds up the first inference
    model(tf.randomNormal([1, 100]));
    
    // return model
    return model
}

// defines the model inference function
async function runModel(){
    
  
    model = loadModel
    
    // gets model prediction
    y = model(tf.randomNormal([1, 100]));

    console.log(y.shape);
    
  }