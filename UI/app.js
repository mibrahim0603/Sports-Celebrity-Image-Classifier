Dropzone.autoDiscover = false;

function init() {
    let dz = new Dropzone("#dropzone", {
        url: "#", // FIXED: Prevents default fallback post requests targeting static resources
        maxFiles: 1,
        addRemoveLinks: true,
        dictDefaultMessage: "Drop files here or click to upload",
        autoProcessQueue: false
    });

    dz.on("addedfile", function (file) {
        // Enforce singular layout stack queues
        if (dz.files[1] != null) {
            dz.removeFile(dz.files[0]);
        }
    });

    dz.on("complete", function (file) {
        if (!file.dataURL) return;

        let url = "http://127.0.0.1:5000/classify_image";

        // Transmit base64 image strings directly over targeted HTTP port mapping channels
        $.post(url, {
            image_data: file.dataURL
        }, function (data, status) {
            console.log("Server Response Data Object:", data);

            if (!data || data.length === 0) {
                $("#resultHolder").hide();
                $("#divClassTable").hide();
                $("#error").show();
                return;
            }

            let match = null;
            let bestScore = -1;

            // Extract highest predictive target index match 
            for (let i = 0; i < data.length; ++i) {
                let maxScoreForThisClass = Math.max(...data[i].class_probability);
                if (maxScoreForThisClass > bestScore) {
                    match = data[i];
                    bestScore = maxScoreForThisClass;
                }
            }

            if (match) {
                $("#error").hide();
                $("#resultHolder").show();
                $("#divClassTable").show();

                // Sanitize dataset tracking naming typos cleanly inside the frontend lookup logic
                let predictedClass = match.class;
                if (predictedClass === "viral_kohli") predictedClass = "virat_kohli";
                if (predictedClass === "roder_federer") predictedClass = "roger_federer";

                // Injects matched athlete's card layout directly into your results panel
                let playerCardHtml = $(`[data-player="${predictedClass}"]`).html();
                $("#resultHolder").html(playerCardHtml);

                // Populate metrics scores tables view cells dynamically
                let classDictionary = match.class_dictionary;
                for (let personName in classDictionary) {
                    let index = classDictionary[personName];
                    let probabilityScore = match.class_probability[index];
                    let elementName = "#score_" + personName;
                    $(elementName).html(probabilityScore + "%");
                }
            }
        });
    });

    $("#submitBtn").on('click', function (e) {
        if (dz.files.length > 0) {
            // Fires event stream workflow loop leading to dz.on("complete") trigger execution
            dz.emit("complete", dz.files[0]);
        }
    });
}

$(document).ready(function () {
    console.log("ready!");
    $("#error").hide();
    $("#resultHolder").hide();
    $("#divClassTable").hide();
    init();
});