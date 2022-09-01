window.onload = () => {
	$('#sendbutton').click(() => {
		console.log("Detectando....")
		$("#detecting").css("display", "block");
		$.ajax({
			url: "/webcam/detections", // fix this to your liking
			type:"POST",
			// data: formData,
			cache: false,
			processData:false,
			contentType:false,
			error: function(data){
				console.log("upload error" , data);
				console.log(data.getAllResponseHeaders());
			},
			success: function(){
				let file = "webcam.mp4"; 
				// file = file.replace("mp4","avi")
				let csv = file
				csv = csv.replace("mp4","csv")
				videobox.attr('src', '..//static//detections//' + file);
				$("#link").css("display", "block");
				$("#download").attr("href", '..//static//detections//' + file);
				$("#csv").attr("href", '..//static//detections//' + csv);
				$("#detecting").css("display", "none");
				console.log("Fin deteccion")
			}
		});
	});
};



// function readUrl(input){
// 	videobox = $('#videobox')
// 	console.log("evoked readUrl")
// 	if(input.files && input.files[0]){
// 		let reader = new FileReader();
// 		reader.onload = function(e){		
// 			videobox.attr('src',e.target.result); 
// 			videobox.height(500);
// 			videobox.width(800);
// 		}
// 		reader.readAsDataURL(input.files[0]);
// 	}

	
// }