window.onload = () => {

	$(document).ready(function(){              
		console.log($("#hasUsed").val())  
		if ($('#hasUsed').val() == 'True'){
			let file = $('#file').val();
			$("#link").css("display", "block");
			$("#download").attr("href", '..//static//detections//' + file + '.avi');
			$("#csv").attr("href", '..//static//detections//' + file + '.csv');
		};    	
	});
}

function hiddenLinks(){
	if ($("#hasUsed") == 'True'){
		$("#link").css("display", "none");
	}
}