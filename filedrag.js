/*
filedrag.js - HTML5 File Drag & Drop demonstration
Featured on SitePoint.com
Developed by Craig Buckler (@craigbuckler) of OptimalWorks.net
*/
(function() {
	
	/******************************/
	// Host and port configuration //
	
	//HOST = 'neuralmechanics.ai/abscbn-api';
	
	HOST = 'http://127.0.0.1:5000';
	
	// URL Calls to flask server
	
	URL_IMAGE = HOST  + '/image/';
	URL_IMAGE_EXT = HOST +'/image_ext/';
	URL_LDA = HOST + '/lda/';
	
	
	
	/******************************/
	var output_data = [];
	var foamtree;
	
	
	// getElementById
	function $id(id) {
		return document.getElementById(id);
	}


	// output information
	function Output(type, msg) {
		
		
		var m = (type == 'image' ? $id("image_div") : $id('text_div') );
		m.innerHTML = msg + m.innerHTML;
	}


	// file drag hover
	function FileDragHover(e) {
		e.stopPropagation();
		e.preventDefault();
		e.target.className = (e.type == "dragover" ? "hover" : "");
	}


	// file selection
	function FileSelectHandler(e) {
		console.log("You dropped the file!");
		$id("alert").style="display: none;";
				
		// cancel event and hover styling
		FileDragHover(e);
		
		// dragged image from HTML page
		if (typeof e.dataTransfer != 'undefined' && e.dataTransfer.files.length==0){
			
			console.log('received image from outside!');
			concepts = SendFlaskRequest(e.dataTransfer.getData('text'));
			DisplayKeywords(concepts, e.dataTransfer.getData('text') )
			
		} 
		
		else{
			console.log("You dropped the file! and we are parsing it!");
		// fetch FileList object
		var files = e.target.files || e.dataTransfer.files;
		
		// process all File objects
		for (var i = 0, f; f = files[i]; i++) {			
			console.log(f);
			fileTypeChecker(f);
			ParseFile(f);
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
		// $id("loading").style.display = "none";
		return null
	}

	
	function resetValues(){
		$id("image_div").innerHTML = "";
		$id("text_div").innerHTML = ""
		
		output_data = [];
	
	}
	
	function analyzeValues(){
	try{
		if(output_data.length == 0) {$id("alert").style="display: visible;";
		return;}
		
		$id("myModal").style.visibility = "visible";
		
		json_data = ((JSON.stringify((output_data))));
		
		var flask_request = new XMLHttpRequest();		
	
		// check if data contains image bytes or image URL
		var URL = URL_LDA + btoa(json_data);
		
		
		console.log('input: ' + json_data);
		
		flask_request.open("GET", URL, false);
		flask_request.send(null);
				

		groups = JSON.parse(flask_request.responseText)
		
		console.log(groups)
		  
		  // Some data to visualize.
          foamtree.set({dataObject: { 
		  'groups': groups.data
 }
 


          });
	
	foamtree.redraw(); } catch(err){ 
		
		console.log(err);
		$id("alert").style="display: visible;";
		$id("myModal").style.visibility = "hidden";
		
		}
	}
	
	function DisplayKeywords(concepts, data){
			
			concept_string = "";
			
			for (var i = 0; i < concepts.length ; i++) {concept_string = concept_string + concepts[i].name + '<br>';}
				
			keywords = (concepts.length == 0 ? '' : (concepts[0].name + ', ' + concepts[1].name + ', ' + 
					concepts[2].name));
			
			output_data.push(concept_string.replace(/<br>/g, ',')
				.replace('no person', ''));
			console.log(output_data);
			
			Output( type='image',
			
			//	"<p><strong>" + file.name + ":</strong><br />" +
				"<div class='imgContainer'>" + 
				'<img src="' + data + '" height= 230 width= 230/><br>' +
				'<p><i>' +  keywords + '</p></i>' + 
				'<div class="overlay">' + 
				'<b>Keywords:</b> <br>' + concept_string +
				'</div>' +
				"</div>"
			);
			
		
	}
	
	function SendFlaskRequest(data){
		
		try{
		
		data = data.replace(/^data:image\/[a-z]+;base64,/, "");
		
		console.log('flask input: ' + data);
		var flask_request = new XMLHttpRequest();
		
		
		var input = ""+(btoa(decodeURIComponent(data)));
		
		// check if data contains image bytes or image URL
		var URL = (data.substring(0,4) == 'http') ? URL_IMAGE_EXT + input : URL_IMAGE + input;
		

		console.log('Running ClarifAI API in flask ...');
		console.log('URL: ' + URL);
		console.log('input: ' + input);
		
		flask_request.open("GET", URL, false);
		flask_request.send(null);
				
		concepts = JSON.parse(JSON.parse(flask_request.responseText).data).outputs[0].data.concepts;
		
		console.log(concepts);} catch(err){ concepts = []; 
		
		console.log(err);
		$id("alert").style="display: visible;";
		}
		
		return(concepts);
		
	}
	// output file information
	function ParseFile(file) {
		
		/*Output(
			"<p>File information: <strong>" + file.name +
			"</strong> type: <strong>" + file.type +
			"</strong> size: <strong>" + file.size +
			"</strong> bytes</p>"
		);*/

		// display an image
		if (file.type.indexOf("image") == 0) {
			
			// read json files for keywords
			
			var KEYWORD_DIR = "assets/data/keywords/"
			
			var reader = new FileReader();
					
			reader.onload = function(e) {
					
			var key_file = KEYWORD_DIR + file.name.substr(0, file.name.lastIndexOf('.')) + '.json' || file.name;
			
			var request = new XMLHttpRequest();
			
			var concept_string = '';
			
			var concepts = [];
					
		try{
				request.open("GET", key_file, false);
	
			
			// load key file concepts into image
				request.send(null);
		
				returnValue = request.responseText;
				
			// get CONCEPTS array from json file
				concepts = (JSON.parse(JSON.parse(returnValue)).outputs[0].data.concepts);
				

			} catch(err){
				
				console.log(e.target.result);
				
				console.log('Error encountered in reading concept json - ' + err.message + ' : ' + file.name);
						
				concepts = SendFlaskRequest(e.target.result);
			}
			
				DisplayKeywords(concepts, e.target.result);
		}

			
			
			reader.readAsDataURL(file);
		}

		// display text
		if (file.type.indexOf("text") == 0) {
			
			var reader = new FileReader();
			
			
			reader.onload = function(e) {
				
			/**********************************/
			
			encoded_text = encodeURIComponent(e.target.result)
			json_data = ((JSON.stringify([encoded_text])));
		
			var flask_request = new XMLHttpRequest();		
	
			// check if data contains image bytes or image URL
			var URL = URL_LDA + btoa(json_data);
		
		
			console.log('input: ' + json_data);
		
			flask_request.open("GET", URL, false);
			flask_request.send(null);
				

			keywords = JSON.parse(flask_request.responseText);
			keywords_list = "";
			
			for(ctr=0; ctr < keywords.data.length; ctr++){
				
				keywords_list = keywords_list + " " + (keywords.data[ctr].label);
				
			}
				
			/***********************************/
				
				output_data.push(keywords_list);
				console.log(output_data); 	
				
				Output( type='text',
					//"<p><strong>" + file.name + ":</strong></p> +
					"<div class='textContainer'>" + 
					e.target.result.replace(/</g, "&lt;").replace(/>/g, "&gt;") +
					"</div>"
				);
			}
			reader.readAsText(file);
		}

	}

	


	// initialize
	function Init() {
		
		// Get the modal
		var modal = $id('myModal');


		var fileselect = $id("fileselect"),
			filedrag = $id("filedrag")

		// file select
		fileselect.addEventListener("change", FileSelectHandler, false);
		
	
		// When user clicks anywhere outside of the modal, close it
		window.onclick = function(event) {
			if (event.target == modal) {
				modal.style.visibility = "hidden";			
			}
		}

		// is XHR2 available?
		var xhr = new XMLHttpRequest();
		if (xhr.upload) {

			// file drop
			filedrag.addEventListener("dragover", FileDragHover, false);
			filedrag.addEventListener("dragleave", FileDragHover, false);
			filedrag.addEventListener("drop", 	FileSelectHandler, false);
		
			filedrag.style.display = "block";

	
		}
		
		/******************************************************************/		
		foamtree = new CarrotSearchFoamTree({
          // Identifier of the HTML element defined above
          id: "modal_content",
		  
		  pixelRatio: window.devicePixelRatio || 1,

          // Remove restriction on the minimum group diameter, so that
          // we can render as many diagram levels as possible.
          groupMinDiameter: 0,

          // Lower the minimum label font size a bit to show more labels.
          groupLabelMinFontSize: 3,

          // Lower the border radius a bit to fit more groups.
          groupBorderWidth: 1.5,
		  
          groupInsetWidth: 5,
		  
          groupSelectionOutlineWidth: 5,

        });
		
			        // Resize FoamTree on orientation change
        window.addEventListener("orientationchange", foamtree.resize);

        // Resize on window size changes
        window.addEventListener("resize", (function() {
          var timeout;
          return function() {
            window.clearTimeout(timeout);
            timeout = window.setTimeout(foamtree.resize, 300);
          }
        })());


/******************************************************************/


	}

	// call initialization file
	if (window.File && window.FileList && window.FileReader) {
		
		Init();
		

	}


})();