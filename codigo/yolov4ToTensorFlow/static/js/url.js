window.onload = () => {
	$('#sendbutton').click(() => {
		imagebox = $('#imagebox')
		console.log($('#imageinput'))
		input = $('#imageinput')[0]
		if(input.value)
		{
			let formData = new FormData();
			formData.append('images' , input.value);
			$.ajax({
				url: "/image_url", 
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				// error: function(data){
				// 	console.log("upload error" , data);
				// 	console.log(data.getAllResponseHeaders());
				// },
				error: function(xhr, status, error) {
					var err = xhr.responseText.split('<p>').pop()
					Swal.fire({
						text: err.replace('</p>', ''),
						icon: 'error',
						confirmButtonText: 'OK',
					})
				},
				success: function(data){
					let file = data['response'][0].image
					let csv = file
					csv = csv.replace("png","csv")
					$("#imagebox").css("display", "block");
					imagebox.attr('src' , '..//static//detections//' + file + '?rand=' + Math.random());
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
	$('#sendbutton').css("display", "block")
}