window.onload = () => {
	$('#sendbutton').click(() => {
		imagebox = $('#imagebox')
		input = $('#imageinput')[0]
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
					console.log(data)
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());
				},
				success: function(data){
					let filename = $('input[type=file]').val().split('\\').pop();
					console.log(filename)
					let tarr = filename.split('/');   
					let file = tarr[tarr.length-1]; 
					file = file.replace("mp4","avi")
					let csv = file
					csv = csv.replace("avi","csv")
					imagebox.attr('src' , '..//static//detections//' + file);
					$("#link").css("display", "block");
         			$("#download").attr("href", '..//static//detections//' + file);
					$("#csv").attr("href", '..//static//detections//' + csv);
				}
			});
		}
	});
};



function readUrl(input){
	imagebox = $('#imagebox')
	console.log("evoked readUrl")
	if(input.files && input.files[0]){
		let reader = new FileReader();
		reader.onload = function(e){		
			imagebox.attr('src',e.target.result); 
			imagebox.height(500);
			imagebox.width(800);
		}
		reader.readAsDataURL(input.files[0]);
	}

	
}