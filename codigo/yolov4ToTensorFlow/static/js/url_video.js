window.onload = () => {
	$('#sendbutton').click(() => {
		$("#detecting").css("display", "block");
		imagebox = $('#imagebox')
		input = $('#imageinput').val();
		if(input)
		{
			let formData = new FormData();
			formData.append('videos' , input);
			$.ajax({
				url: "/video_url", 
				type:"POST",
				data: formData,
				cache: false,
				processData: false,
				contentType: false,
				error: function(data){
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());
					$("#detecting").css("display", "none");
				},
				success: function(data){
					let file = data['response'][0].video
					let csv = file
					csv = csv.replace(".mp4",".csv")
					$("#imagebox").css("display", "block");
					imagebox.attr('src' , '..//static//detections//' + file);
					$("#link").css("display", "block");
         			$("#download").attr("href", '..//static//detections//' + file);
					$("#csv").attr("href", '..//static//detections//' + csv);
					$("#detecting").css("display", "none");
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
	$('#sendbutton').css("display", "block")
}