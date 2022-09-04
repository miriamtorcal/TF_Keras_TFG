window.onload = () => {
	$('#sendbutton').click(() => {
		imagebox = $('#imagebox')
		input = $('#imageinput1')[0]
		if(input.files && input.files[0])
		{
			let formData = new FormData();
			formData.append('images' , input.files[0]);
			$.ajax({
				url: "/image/detections", 
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				error: function(xhr, status, error) {
					var err = xhr.responseText.split('<p>').pop()
					Swal.fire({
						text: err.replace('</p>', ''),
						icon: 'error',
						confirmButtonText: 'OK',
					})
				},
				success: function(data){
					let filename = $('input[type=file]').val().split('\\').pop();
					console.log(filename)
					let tarr = filename.split('/');   
					let file = tarr[tarr.length-1]; 
					file = file.replace("jpg","png")
					let csv = file
					csv = csv.replace("png","csv")
					imagebox.attr('src' , 'static/detections/' + file + '?rand=' + Math.random());
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
	console.log(input.files[0])
	$('#msg_span').text(input.files[0].name)
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