using System;

class Program
{
    static void Main()
    {
        Console.WriteLine("Enter text to predict sentiment");
        var textToPredict = Console.ReadLine() ?? string.Empty;

        var sentimentService = new SentimentService("sentiment.csv");
        var prediction = sentimentService.Predict(textToPredict);

        PrintPrediction(textToPredict, prediction);
    }

    private static void PrintPrediction(string input, SentimentPrediction prediction)
    {
        Console.WriteLine($"Text: {input}");
        Console.WriteLine($"Prediction: {(prediction.PredictedLabel ? "Positive" : "Negative")}, Probability: {prediction.Probability:P2}");
    }
}