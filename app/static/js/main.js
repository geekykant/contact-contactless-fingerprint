$("#predict_submit").click(e => {
  e.preventDefault();
  sendFileToPredictor();
});

const validateFiles = () => {
  return $('#fp1').prop('files').length != 0 && $('#fp2').prop('files').length != 0
}

function sendFileToPredictor() {
  if (!validateFiles()) {
    alert("You haven't inputted 2 images!");
    return;
  }

  var formData = new FormData();
  const fp1 = $('#fp1').prop('files')[0];
  const fp2 = $('#fp2').prop('files')[0];
  formData.append('fp1', fp1, fp1.name);
  formData.append('fp2', fp2, fp2.name);

  $.ajax({
    type: "POST",
    url: "/",
    data: formData,
    processData: false,
    contentType: false,
    success: function(result) {
      console.log(result);
      const accuracy = result['accuracy'];
      if(accuracy > 85){
        $('#pred_emoji').attr('x-text',"`👍`");
      }else{
          $('#pred_emoji').attr('x-text', "`❌`");
      }

      $('#accuracy_text').text(`${accuracy}%`);
      $("#accuracy_circle").attr("x-data",`{ circumference: 50 * 2 * Math.PI, percent: ${Math.round(accuracy)} }`);
    },
    error: function(err) {
      alert(err);
      console.log(err);
    }
  });
}
