window.onload = () => {
	$('#sendbutton').click(() => {
		imagebox = $('#imagebox')
		input = $('#imageinput1')[0]
		folder = $('#folder_name').val()
		if(input.files && input.files[0])
		{
			let formData = new FormData();
			formData.append('files' , input.files[0]);
			formData.append('folder_name' , folder);
			console.log(input.files[0].name)
			let ext = input.files[0].name.split('.').pop()
			console.log(ext)
			if(ext === 'names'){
				$.ajax({
					url: "/names_add", 
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
						Swal.fire({
							text: 'Fichero subido correctamente',
							icon: 'success',
							confirmButtonText: 'OK',
						})
						.then(function() {
							console.log('pre reset input')
							$('#msg_span').text('Escoge un fichero de nombres')
							$('#imageinput1').val('')
							$('#sendbutton').css("display", "none")
						})
					}
				})
			}
			else if(ext === 'pb' || ext === 'tflite'){
				$.ajax({
					url: "/model_add", 
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
						Swal.fire({
							text: 'Fichero subido correctamente',
							icon: 'success',
							confirmButtonText: 'OK',
						})
						.then(function() {
							console.log('pre reset input')
							$('#msg_span').text('Escoge un modelo')
							$('#imageinput1').val('')
							$('#folder_name').val('')
							$('#sendbutton').css("display", "none")
						})
					}
				})
			}
		}
	});
};



function readUrl(input){
	imagebox = $('#imagebox')
	console.log(input.files)
	$('#msg_span').text(input.files[0].name)
	if(input.files && input.files[0]){
		let reader = new FileReader();
		reader.readAsDataURL(input.files[0]);
	}
	$('#sendbutton').css("display", "block")
}