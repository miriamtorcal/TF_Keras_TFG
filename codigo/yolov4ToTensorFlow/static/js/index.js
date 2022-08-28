window.onload = () => {
	$('#sendbutton').click(() => {
		input = $('#imageinput1')[0]
		folder = $('#folder_name').val()
		select = $('#file_selected').val()
		model = $('#model_selected').val()

		if(input != undefined){
			if(input.files && input.files[0])
			{
				let formData = new FormData();
				if (input.files.length == 1){
					formData.append('files' , input.files[0]);
				}
				else if(input.files.length > 1){
					for (let i = 0; i < input.files.length; i++) {
						formData.append('files' , input.files[i]);
					}
				}
				formData.append('folder_name' , folder);
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
		}
		if(select != undefined)
		{
			let formData = new FormData();
			formData.append('new_file', select)
			$.ajax({
				url: "/names_file", 
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				error: function(data){
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());
				},
				success: function(data){
					Swal.fire({
						text: 'Fichero de nombres cambiado correctamente',
						icon: 'success',
						confirmButtonText: 'OK',
					})
					.then(function() {
						console.log(data.new_file_selected)
						$('#file_selected').val(data.new_file_selected)
					})
				}
			})
		}
		if(model != undefined)
		{
			let formData = new FormData();
			formData.append('new_file', model)
			$.ajax({
				url: "/model_file", 
				type:"POST",
				data: formData,
				cache: false,
				processData:false,
				contentType:false,
				error: function(data){
					console.log("upload error" , data);
					console.log(data.getAllResponseHeaders());
				},
				success: function(data){
					Swal.fire({
						text: 'Modelo de deteccion cambiado correctamente',
						icon: 'success',
						confirmButtonText: 'OK',
					})
					.then(function() {
						console.log(data.new_file_selected)
						$('#model_selected').val(data.new_file_selected)
					})
				}
			})
		}
	});
};



function readUrl(input){
	console.log(input.files)
	if (input.files.length === 1){
		$('#msg_span').text(input.files[0].name)
	}
	else if(input.files.length > 1){
		$('#msg_span').text(input.files.length + " archivos seleccionados")
	}
	
	if(input.files && input.files[0]){
		let reader = new FileReader();
		reader.readAsDataURL(input.files[0]);
	}
	$('#sendbutton').css("display", "block")
}