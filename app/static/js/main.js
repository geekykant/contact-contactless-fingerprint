$(function() {
  $(document).ready(function() {
    const crop_format = {
      enableExif: true,
      viewport: {
        width: 160,
        height: 160,
        type: 'circle' //circle
      },
      boundary: {
        width: 200,
        height: 200
      }
    };

    $image_crop1 = $('#f1').croppie(crop_format);
    $image_crop2 = $('#f2').croppie(crop_format);

    $('#fp1_upload').click(() => {
      $('#fp1').click();
    });
    $('#fp2_upload').click(() => {
      $('#fp2').click();
    });

    let sample_check1 = false, sample_check2 = false

    $('#fp1_sample').click(() => {
      $image_crop1.croppie('bind', {
        url: '/static/img/sample_fingerprint.png',
        zoom: 0
      }).then(function() {
        console.log('sample finger 1 added');
        sample_check1 = true
      });
    });

    $('#fp2_sample').click(() => {
      $image_crop2.croppie('bind', {
        url: '/static/img/sample_fingerprint.png',
        zoom: 0
      }).then(function() {
        console.log('sample finger 2 added');
        sample_check2 = true
      });
    });

    $('#fp1').on('change', function() {
      var reader = new FileReader();
      reader.onload = function(event) {
        $image_crop1.croppie('bind', {
          url: event.target.result,
          zoom: 0
        }).then(function() {
          console.log('jQuery bind 1 complete');
        });
      }
      reader.readAsDataURL(this.files[0]);
    });

    $('#fp2').on('change', function() {
      var reader = new FileReader();
      reader.onload = function(event) {
        $image_crop2.croppie('bind', {
          url: event.target.result,
          zoom: 0
        }).then(function() {
          console.log('jQuery bind 2 complete');
        });
      }
      reader.readAsDataURL(this.files[0]);
    });

    function makeblob(dataURL) {
        const BASE64_MARKER = ';base64,';
        const parts = dataURL.split(BASE64_MARKER);
        const contentType = parts[0].split(':')[1];
        const raw = window.atob(parts[1]);
        const rawLength = raw.length;
        const uInt8Array = new Uint8Array(rawLength);

        for (let i = 0; i < rawLength; ++i) {
            uInt8Array[i] = raw.charCodeAt(i);
        }

        return new Blob([uInt8Array], { type: contentType });
    }

    $("#predict_submit").click(e => {
      e.preventDefault();
      sendFileToPredictor();
    });

    const validateFiles = () => {
      return $('#fp1').prop('files').length != 0 && $('#fp2').prop('files').length != 0
    }

    const validateSampleFiles = () => {
      return sample_check1 && sample_check2
    }

    function sendFileToPredictor() {
      if (!validateFiles() && !validateSampleFiles()) {
        alert("You haven't inputted 2 images!");
        return;
      }

      var formData = new FormData();
      // const fp1 = $('#fp1').prop('files')[0];
      // const fp2 = $('#fp2').prop('files')[0];
      // formData.append('fp1', fp1, fp1.name);
      // formData.append('fp2', fp2, fp2.name);

      var fp1_cropped,
        fp2_cropped;

      $image_crop1.croppie('result', {
        type: 'canvas',
        size: 'viewport'
      }).then((crop_img1) => {
        fp1_cropped = makeblob(crop_img1);
        $image_crop2.croppie('result', {
          type: 'canvas',
          size: 'viewport'
        }).then((crop_img2) => {
          fp2_cropped = makeblob(crop_img2);
        }).then(() => {
          formData.append('fp1', fp1_cropped);
          formData.append('fp2', fp2_cropped);

          $.ajax({
            type: "POST",
            url: "/",
            data: formData,
            processData: false,
            contentType: false,
            success: function(result) {
              // console.log(result);
              const accuracy = result['accuracy'];
              if (accuracy > 85) {
                $('#pred_emoji').attr('x-text', "`👍`");
              } else {
                $('#pred_emoji').attr('x-text', "`❌`");
              }

              $('#accuracy_text').text(`${accuracy}%`);
              $("#accuracy_circle").attr("x-data", `{ circumference: 50 * 2 * Math.PI, percent: ${Math.round(accuracy)} }`);
            },
            error: function(err) {
              alert(err);
              console.log(err);
            }
          });
        })
      });

    }
  });
});
