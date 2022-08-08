window.onload = () => {
	$('#sendbutton').click(() => {
		console.log("Detectando....")
		$("#detecting").css("display", "block");
		videobox = $('#videobox')
		input = $('#videoinput1')[0]
		if(input.files && input.files[0])
		{
			let formData = new FormData();
			formData.append('videos' , input.files[0]);
			$.ajax({
				url: "/video/detections", // fix this to your liking
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				error: function(data){
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());
				},
				success: function(){
					let filename = $('input[type=file]').val().split('\\').pop();
					console.log(filename)
					let tarr = filename.split('/');   
					let file = tarr[tarr.length-1]; 
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
		}
	});
};



function readUrl(input){
	videobox = $('#videobox')
	console.log("evoked readUrl")
	$('#msg_span').text(input.files[0].name)
	if(input.files && input.files[0]){
		let reader = new FileReader();
		reader.onload = function(e){		
			videobox.attr('src',e.target.result); 
			videobox.height(500);
			videobox.width(800);
		}
		reader.readAsDataURL(input.files[0]);
	}

	
}