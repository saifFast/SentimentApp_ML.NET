using Microsoft.ML;

// 1. Create ML Context
var mlContext = new MLContext();

// 2. Load Data
var dataPath = "sentiment.csv";
var dataView = mlContext.Data.LoadFromTextFile<SentimentData>(
    path: dataPath,
    hasHeader: true,
    separatorChar: ',');

// 3. Define the training pipeline
var pipeline = mlContext.Transforms.Text.FeaturizeText("Features", nameof(SentimentData.Text))
    .Append(mlContext.BinaryClassification.Trainers.SdcaLogisticRegression(
        labelColumnName: nameof(SentimentData.Label), featureColumnName: "Features"));

// 4. Train the model
var model = pipeline.Fit(dataView);

// 5. Create prediction engine
var predEngine = mlContext.Model.CreatePredictionEngine<SentimentData, SentimentPrediction>(model);

// 6. Make a prediction
var input = new SentimentData { Text = "I hate it" };
var prediction = predEngine.Predict(input);

Console.WriteLine($"Text: {input.Text}");
Console.WriteLine($"Prediction: {(prediction.PredictedLabel ? "Positive" : "Negative")}, Probability: {prediction.Probability:P2}");