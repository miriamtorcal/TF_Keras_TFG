window.onload = () => {
	const dragArea = document.querySelector(".input");
	let file; 
	var files = [];
	let fileType;

	if(dragArea != undefined || dragArea != null){

		dragArea.addEventListener('dragover', (event) => {
			event.preventDefault();
		})

		dragArea.addEventListener('dragleave', () => {

		})

		dragArea.addEventListener('drop', async (event) => {
			event.preventDefault();
			file = event.dataTransfer.files[0];
			let items = event.dataTransfer.items;

			fileType = file.name.split('.').pop();
			fileType = file.name == fileType  ? 'folder' : fileType;

			let validExtensions = ['tflite', 'folder']
			if(validExtensions.includes(fileType)){
				if(fileType === 'folder'){
					for (let i=0; i<items.length; i++){
						await verifyItems(items[i].webkitGetAsEntry(), files)
						.then(_ => {
							console.log(_)
						})
					}
					$('#msg_span').text('Cargando los archivos seleccionados')
					$('#sendbutton').css("display", "block")
				}
				else{
					$('#msg_span').text(file.name)
					let fileReader = new FileReader()
					fileReader.readAsDataURL(file);
					$('#sendbutton').css("display", "block")
				}
			}
		}, false)
	}

	$('#sendbutton').click(() => {
		input = $('#imageinput1')[0]
		folder = $('#folder_name').val()
		select = $('#file_selected').val()
		model = $('#model_selected').val()

		$('#circle').css("display", "block")

		if(input != undefined){
			if (input.files.length == 0 && file != undefined){
				let formData = new FormData();
				formData.append('folder_name', folder)
				if(fileType === 'tflite'){
					formData.append('files', file);
					$.ajax({
						url: "/model_add", 
						type:"POST",
						data: formData,
						cache: false,
						processData:false,
						contentType:false,
						// error: function(data){
						// 	
						// 	console.log(data.responseText)
						// 	console.log("upload error" , data);
						// 	console.log(data.getAllResponseHeaders());
						// },
						error: function(xhr, status, error) {
							$('#msg_span').text('Arrastra/selecciona un modelo o carpeta')
							$('#imageinput1').val('')
							$('#folder_name').val('')
							$('#sendbutton').css("display", "none")
							$('#circle').css("display", "none")
							var err = xhr.responseText.split('<p>').pop()
							Swal.fire({
								text: err.replace('</p>', ''),
								icon: 'error',
								confirmButtonText: 'OK',
							})
						},
						success: function(){
							Swal.fire({
								text: 'Fichero subido correctamente',
								icon: 'success',
								confirmButtonText: 'OK',
							})
							.then(function() {
								$('#msg_span').text('Escoge un modelo')
								$('#imageinput1').val('')
								file = ''
								files = []
								$('#folder_name').val('')
								$('#sendbutton').css("display", "none")
								$('#circle').css("display", "none")
							})
						}
					})
				}
				if(fileType === 'folder'){
					$('#msg_span').text(files.length + ' archivos seleccionados')
					for (let i = 0; i < files.length; i++) {
						formData.append('files', files[i])
						
					}
					// formData['files'] = files
					formData.append('files', files)
					formData.append('folder_name', folder)
					$.ajax({
						url: "/model_add", 
						type:"POST",
						data: formData,
						cache: false,
						processData:false,
						contentType:false,
						// error: function(data){
						// 	
						// 	console.log(data.responseText)
						// 	console.log("upload error" , data);
						// 	console.log(data.getAllResponseHeaders());
						// },
						error: function(xhr, status, error) {
							$('#msg_span').text('Arrastra/selecciona un modelo o carpeta')
							$('#imageinput1').val('')
							$('#folder_name').val('')
							$('#sendbutton').css("display", "none")
							$('#circle').css("display", "none")
							var err = xhr.responseText.split('<p>').pop()
							Swal.fire({
								text: err.replace('</p>', ''),
								icon: 'error',
								confirmButtonText: 'OK',
							})
						},
						success: function(){
							Swal.fire({
								text: 'Fichero subido correctamente',
								icon: 'success',
								confirmButtonText: 'OK',
							})
							.then(function() {
								$('#msg_span').text('Arrastra/selecciona un modelo o carpeta')
								$('#imageinput1').val('')
								$('#folder_name').val('')
								$('#sendbutton').css("display", "none")
								$('#circle').css("display", "none")
								file = ''
								files = []
							})
						}
					})
				}
			} 
			else if(input.files && input.files[0])
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
						// error: function(data){
						// 
						// 	console.log(data.responseText)
						// 	console.log("upload error" , data);
						// 	console.log(data.getAllResponseHeaders());
						// },
						error: function(xhr, status, error) {	
							$('#msg_span').text('Arrastra/selecciona un modelo o carpeta')
							$('#imageinput1').val('')
							$('#folder_name').val('')
							$('#sendbutton').css("display", "none")
							$('#circle').css("display", "none")
							var err = xhr.responseText.split('<p>').pop()
							Swal.fire({
								text: err.replace('</p>', ''),
								icon: 'error',
								confirmButtonText: 'OK',
							})
						},
						success: function(){
							Swal.fire({
								text: 'Fichero subido correctamente',
								icon: 'success',
								confirmButtonText: 'OK',
							})
							.then(function() {
								console.log('pre reset input')
								$('#msg_span').text('Escoge un fichero de etiquetas')
								$('#imageinput1').val('')
								$('#sendbutton').css("display", "none")
								$('#circle').css("display", "none")
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
						// error: function(data){
						// 	console.log(data.responseText)
						// 	console.log("upload error" , data);
						// 	console.log(data.getAllResponseHeaders());
						// },
						error: function(xhr, status, error) {
							$('#msg_span').text('Arrastra/selecciona un modelo o carpeta')
							$('#imageinput1').val('')
							$('#folder_name').val('')
							$('#sendbutton').css("display", "none")
							$('#circle').css("display", "none")
							var err = xhr.responseText.split('<p>').pop()
							Swal.fire({
								text: err.replace('</p>', ''),
								icon: 'error',
								confirmButtonText: 'OK',
							})
						},
						success: function(){
							$('#circle').css("display", "none")
							Swal.fire({
								text: 'Fichero subido correctamente',
								icon: 'success',
								confirmButtonText: 'OK',
							})
							.then(function() {
								$('#msg_span').text('Arrastra/selecciona un modelo o carpeta')
								$('#imageinput1').val('')
								$('#folder_name').val('')
								$('#sendbutton').css("display", "none")
								$('#circle').css("display", "none")
							})
						}
					})
				}
				else{
					Swal.fire({
						text: 'Extension no valida',
						icon: 'error',
						confirmButtonText: 'OK'
					})
					.then(function() {
						$('#msg_span').text('Arrastra/selecciona un modelo o carpeta')
						$('#imageinput1').val('')
						$('#folder_name').val('')
						$('#sendbutton').css("display", "none")
						$('#circle').css("display", "none")
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
				// error: function(data){
				// 	console.log(data.responseText)
				// 	console.log("upload error" , data);
				// 	console.log(data.getAllResponseHeaders());
				// },
				error: function(xhr, status, error) {
					$('#circle').css("display", "none")
					var err = xhr.responseText.split('<p>').pop()
					Swal.fire({
						text: err.replace('</p>', ''),
						icon: 'error',
						confirmButtonText: 'OK',
					})
				},
				success: function(data){
					Swal.fire({
						text: 'Fichero de etiquetas cambiado correctamente',
						icon: 'success',
						confirmButtonText: 'OK',
					})
					.then(function() {
						$('#file_selected').val(data.new_file_selected)
						$('#circle').css("display", "none")
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
				// error: function(data){
				// 	console.log(data.responseText)
				// 	console.log("upload error" , data);
				// 	console.log(data.getAllResponseHeaders());
				// },
				error: function(xhr, status, error) {
					$('#circle').css("display", "none")
					var err = xhr.responseText.split('<p>').pop()
					Swal.fire({
						text: err.replace('</p>', ''),
						icon: 'error',
						confirmButtonText: 'OK',
					})
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
						$('#circle').css("display", "none")
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

async function verifyItems(item, files){
	if (item.isDirectory) {
		directoryReader = item.createReader();
		directoryReader.readEntries(async function (entries){
			for (let k=0; k<entries.length; k++){
				f = entries[k]
				if(f.isFile){
					f.file (async function (file) {
						files.push(await file)
					})

				}
				// }
				if(f.isDirectory){
					if (f.name != 'assets'){
						await verifyItems(await f,files)
					}
				}
			}
			return files;
		});
	}
	return files;
}