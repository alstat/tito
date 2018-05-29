/*

filedrag.js - HTML5 File Drag & Drop demonstration
Featured on SitePoint.com
Developed by Craig Buckler (@craigbuckler) of OptimalWorks.net

*/
(function () {
	// - - - - - - - - - - - - - - - - - - - - - //
	// HOST AND PORT
	//HOST = 'neuralmechanics.ai/abscbn-api';

	HOST = 'http://127.0.0.1:5000';

	// URL Calls to flask server
	URL_ZIP_FILE = HOST + "/zip_file"
	URL_TRAIN = HOST + "/train"
	URL_ANALYZE = HOST + "/analyze"
	// - - - - - - - - - - - - - - - - - - - - - //

	var output_data = [];
	var foamtree;

	// getElementById
	function $id(id) {
		return document.getElementById(id);
	}

	// output information
	function output(type, msg) {
		var m = (type == 'image' ? $id("image_div") : $id('text_div'));
		m.innerHTML = msg + m.innerHTML;
	}

	// file drag hover
	function fileDragHover(e) {
		e.stopPropagation();
		e.preventDefault();
		e.target.className = (e.type == "dragover" ? "hover" : "");
	}

	// file selection
	function fileSelectHandler(e) {
		console.log("You dropped the file!");
		$id("alert").style = "display: none;";

		// cancel event and hover styling
		fileDragHover(e);

		// dragged image from HTML page
		if (typeof e.dataTransfer != 'undefined' && e.dataTransfer.files.length === 0) {
			console.log('received image from outside!');
		} else {
			var files = e.target.files || e.dataTransfer.files;

			// process all File objects
			for (var i = 0, f; f = files[i]; i++) {
				console.log(f);
				fileTypeChecker(f);
				var encodedFile = getBase64(f);

				parseFile(f);
			}
		}
	}

	function fileTypeChecker(f) {
		$id("loading").style.display = "block";
		var fileExtension = f["name"].split(".").pop();

		if ($id("select-input-type").value === "Photos (ZIP File)") {
			var expectedExtension = "zip";
		} else if ($id("select-input-type").value === "Videos (MP4, etc.)") {
			var expectedExtension = "mp4";
		}

		if (fileExtension === expectedExtension) {
			console.log("Recieved ZIP file");
		} else if (
			fileExtension === expectedExtension ||
			fileExtension === "mkv" ||
			fileExtension === "3gp"
		) {
			console.log("Received Video");
		} else {
			alert("Expected input " + $id("select-input-type").value + ", but received a " + fileExtension + " file.");
		}
		return null
	}

	function getBase64(f) {
		var reader = new FileReader();
		reader.readAsDataURL(f);
		reader.onload = function () {
			var data = reader.result;

			function callback(output) {
				console.log(output);
				console.log(JSON.parse(output));
			}
			sendRequest(URL_ZIP_FILE, data, callback);
		};
		reader.onerror = function (error) {
			console.log("Error: ", error)
		};
	}
	
	function train() {
		function callback(output) {
			console.log(output);
			console.log(typeof(output));
			console.log(JSON.parse(output));
			$id("accuracy").innerHTML = JSON.parse(output)["accuracy"];
			$id("runtime").innerHTML = JSON.parse(output)["runtime"];
		}
		sendRequest(URL_TRAIN, {}, callback);
	}

	function analyze() {
		function callback(output) {
			console.log(output);
			console.log(typeof(output));
			console.log(JSON.parse(output));
			$id("noOfFrames").innerHTML = JSON.parse(output)["frames"];
			$id("impRuntime").innerHTML = JSON.parse(output)["runtime"];
		}
		sendRequest(URL_ANALYZE, {}, callback);
	}

	function sendRequest(URL, data, callback) {
		var xhr = new XMLHttpRequest();
		xhr.open("POST", URL, true);
		xhr.setRequestHeader("Content-type", "application/x-www-form-urlencoded");
		xhr.send(data);
		xhr.onreadystatechange = function () {
			if (this.readyState === 4 && this.status === 200) {
				var output = this.responseText;
				callback(output);
			}
		}
	}

	// output file information
	function parseFile(file) {
		console.log("Eow pow!");
	}

	// initialize
	function Init() {
		// Get the modal
		var modal = $id('myModal');
		var fileselect = $id("fileselect"),
			filedrag = $id("filedrag")

		// file select
		fileselect.addEventListener("change", fileSelectHandler, false);

		// When user clicks anywhere outside of the modal, close it
		window.onclick = function (event) {
			if (event.target == modal) {
				modal.style.visibility = "hidden";
			}
		}

		// is XHR2 available?
		var xhr = new XMLHttpRequest();
		if (xhr.upload) {
			// file drop
			filedrag.addEventListener("dragover", fileDragHover, false);
			filedrag.addEventListener("dragleave", fileDragHover, false);
			filedrag.addEventListener("drop", fileSelectHandler, false);
			filedrag.style.display = "block";
		}
	}

	// call initialization file
	if (window.File && window.FileList && window.FileReader) {
		Init();
	}

	$id("train-btn").addEventListener("click", function () {
		train();
	})

	$id("analyze-btn").addEventListener("click", function () {
		analyze();
	})
})();