import { ComfyApp, app } from "../../scripts/app.js";

function load_image(str) {
	let base64String = canvas.toDataURL('image/png');
	let img = new Image();
	img.src = base64String;
}

app.registerExtension({
	name: "Comfy.Inspire.img",

	nodeCreated(node, app) {
		if(node.comfyClass == "LoadImage //Inspire") {
			let w = node.widgets.find(obj => obj.name === 'image_data');

			Object.defineProperty(w, 'value', {
				set(v) {
					if(v != '[IMAGE DATA]')
						w._value = v;
				},
				get() {
					const stackTrace = new Error().stack;
					if(!stackTrace.includes('draw') && !stackTrace.includes('graphToPrompt') && stackTrace.includes('app.js')) {
						return "[IMAGE DATA]";
					}
					else {
						return w._value;
					}
				}
			});

			Object.defineProperty(node, 'imgs', {
				set(v) {
					this._img = v;

					var canvas = document.createElement('canvas');
					canvas.width = v[0].width;
					canvas.height = v[0].height;

					var context = canvas.getContext('2d');
					context.drawImage(v[0], 0, 0, v[0].width, v[0].height);

					var base64Image = canvas.toDataURL('image/png');
					w.value = base64Image;
				},
				get() {
					if(this._img == undefined && w.value != '') {
						this._img = [new Image()];
						if(w.value && w.value != '[IMAGE DATA]')
							this._img[0].src = w.value;
					}

					return this._img;
				}
			});
		}
    }
})