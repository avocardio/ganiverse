let obj
let tex

function preload() {
	obj = loadModel('mesh.stl', true)
    tex = loadImage('generated_image.jpg')
}

function setup() {
	let cvn = createCanvas(300, 300, WEBGL)
	cvn.parent('container')
	noFill()
}

function draw() {
	background(255, 255, 255)
	orbitControl(5)
    // transform the model rotate on its own axis
    rotateX(frameCount * 0.01)
    rotateY(frameCount * 0.01)
    rotateZ(frameCount * 0.01)
    
    texture(tex)

	rotateY(radians(-frameCount / 2))
	model(obj)
}