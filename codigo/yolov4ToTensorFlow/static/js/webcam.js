window.onload = () => {

	$(document).ready(function(){              
		if ($('#hasUsed').val() == 'True'){
			if ($('#csvOnly').val() == 'True'){
				let file = $('#file').val();
				$("#link").css("display", "block");
				$("#download").css("display", "none");
				$("#csv").attr("href", '..//static//detections//' + file + '.csv');
			}
			else{
				let file = $('#file').val();
				$("#link").css("display", "block");
				$("#download").attr("href", '..//static//detections//' + file + '.avi');
				$("#csv").attr("href", '..//static//detections//' + file + '.csv');
			}
		};    	
	});
}

function hiddenLinks(){
	if ($("#hasUsed") == 'True'){
		$("#link").css("display", "none");
	}
}