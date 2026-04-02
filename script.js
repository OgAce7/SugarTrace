// script.js - SugarTrace Frontend Logic
// Handles form submission, API calls and showing results

// Example patient data to demo the form quickly
var exampleData = {
    Pregnancies: 3,
    Glucose: 148,
    BloodPressure: 72,
    SkinThickness: 35,
    Insulin: 125,
    BMI: 33.6,
    DiabetesPedigreeFunction: 0.627,
    Age: 50
};

// Fill the form with example data
function fillExample() {
    document.getElementById('Pregnancies').value = exampleData.Pregnancies;
    document.getElementById('Glucose').value = exampleData.Glucose;
    document.getElementById('BloodPressure').value = exampleData.BloodPressure;
    document.getElementById('SkinThickness').value = exampleData.SkinThickness;
    document.getElementById('Insulin').value = exampleData.Insulin;
    document.getElementById('BMI').value = exampleData.BMI;
    document.getElementById('DiabetesPedigreeFunction').value = exampleData.DiabetesPedigreeFunction;
    document.getElementById('Age').value = exampleData.Age;
}

// Clear all form fields
function clearForm() {
    var fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'];
    for (var i = 0; i < fields.length; i++) {
        document.getElementById(fields[i]).value = '';
    }
    // also hide result boxes
    document.getElementById('resultBox').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';
}

// Collect form data and send to /predict endpoint
function predict() {
    // hide previous results
    document.getElementById('resultBox').style.display = 'none';
    document.getElementById('errorBox').style.display = 'none';

    // get selected model
    var selectedModel = document.getElementById('modelSelect').value;

    // read all input values
    var inputData = {
        model: selectedModel,
        Pregnancies: parseFloat(document.getElementById('Pregnancies').value),
        Glucose: parseFloat(document.getElementById('Glucose').value),
        BloodPressure: parseFloat(document.getElementById('BloodPressure').value),
        SkinThickness: parseFloat(document.getElementById('SkinThickness').value),
        Insulin: parseFloat(document.getElementById('Insulin').value),
        BMI: parseFloat(document.getElementById('BMI').value),
        DiabetesPedigreeFunction: parseFloat(document.getElementById('DiabetesPedigreeFunction').value),
        Age: parseFloat(document.getElementById('Age').value)
    };

    // simple validation - check if any value is missing or NaN
    var fields = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
                  'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'];
    for (var i = 0; i < fields.length; i++) {
        if (isNaN(inputData[fields[i]])) {
            showError('Please fill in all fields. Missing: ' + fields[i]);
            return;
        }
    }

    // send POST request to backend
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(inputData)
    })
    .then(function(response) {
        return response.json();
    })
    .then(function(data) {
        if (data.error) {
            showError(data.error);
        } else {
            showResult(data);
        }
    })
    .catch(function(err) {
        showError('Could not connect to server. Make sure app.py is running. Error: ' + err.message);
    });
}


// Display the prediction result
function showResult(data) {
    var resultBox = document.getElementById('resultBox');
    resultBox.style.display = 'block';

    // set result label
    var resultLabel = document.getElementById('resultLabel');
    if (data.prediction === 1) {
        resultLabel.textContent = '⚠️ Diabetic';
        resultLabel.className = 'result-label diabetic';
    } else {
        resultLabel.textContent = '✅ Not Diabetic';
        resultLabel.className = 'result-label not-diabetic';
    }

    // probability score
    document.getElementById('probScore').textContent = data.probability + '%';

    // risk level
    var riskEl = document.getElementById('riskLevel');
    riskEl.textContent = data.risk_level;
    if (data.risk_level === 'High') {
        riskEl.style.color = '#c0392b';
    } else if (data.risk_level === 'Moderate') {
        riskEl.style.color = '#e67e22';
    } else {
        riskEl.style.color = '#27ae60';
    }

    // model name
    document.getElementById('modelUsed').textContent = data.model_used;

    // update risk bar
    var barFill = document.getElementById('riskBar');
    barFill.style.width = data.probability + '%';
    document.getElementById('riskBarLabel').textContent = data.probability + '% chance of diabetes';

    // feature importance (only for random forest)
    var featSection = document.getElementById('featureImportanceSection');
    var featList = document.getElementById('featureImportanceList');
    featList.innerHTML = '';

    if (data.feature_importance && Object.keys(data.feature_importance).length > 0) {
        featSection.style.display = 'block';

        // sort features by importance
        var items = Object.entries(data.feature_importance);
        items.sort(function(a, b) { return b[1] - a[1]; });

        // find max importance for scaling bars
        var maxVal = items[0][1];

        for (var i = 0; i < items.length; i++) {
            var name = items[i][0];
            var val = items[i][1];
            var barWidth = Math.round((val / maxVal) * 100);

            var row = document.createElement('div');
            row.className = 'importance-row';
            row.innerHTML =
                '<span class="feat-name">' + name + '</span>' +
                '<div class="importance-bar-bg">' +
                    '<div class="importance-bar-fill" style="width:' + barWidth + '%"></div>' +
                '</div>' +
                '<span class="importance-val">' + val.toFixed(3) + '</span>';
            featList.appendChild(row);
        }
    } else {
        featSection.style.display = 'none';
    }

    // scroll to result
    resultBox.scrollIntoView({ behavior: 'smooth' });
}


// Load model performance info
function loadModelInfo() {
    var infoBox = document.getElementById('modelInfoBox');
    var content = document.getElementById('modelInfoContent');

    // toggle visibility
    if (infoBox.style.display === 'block') {
        infoBox.style.display = 'none';
        return;
    }

    infoBox.style.display = 'block';
    content.textContent = 'Loading...';

    fetch('/model-info')
    .then(function(res) { return res.json(); })
    .then(function(data) {
        if (data.error) {
            content.textContent = 'Error: ' + data.error;
            return;
        }

        // build a simple table
        var html = '<table class="stats-table"><thead><tr><th>Model</th><th>Accuracy (%)</th><th>Recall (%)</th><th>ROC-AUC (%)</th></tr></thead><tbody>';
        for (var modelName in data) {
            var m = data[modelName];
            // convert underscore to space for readability
            var displayName = modelName.replace(/_/g, ' ');
            html += '<tr>' +
                '<td>' + displayName + '</td>' +
                '<td>' + m.accuracy + '</td>' +
                '<td>' + m.recall + '</td>' +
                '<td>' + m.roc_auc + '</td>' +
            '</tr>';
        }
        html += '</tbody></table>';
        content.innerHTML = html;
    })
    .catch(function(err) {
        content.textContent = 'Could not load. Server might not be running.';
    });
}


// Show error message
function showError(msg) {
    var errorBox = document.getElementById('errorBox');
    document.getElementById('errorMsg').textContent = msg;
    errorBox.style.display = 'block';
    errorBox.scrollIntoView({ behavior: 'smooth' });
}
