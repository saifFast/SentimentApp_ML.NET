using Microsoft.ML.Data;

public class SentimentData
{
    [LoadColumn(0)]
    public string? Text { get; set; }

    [LoadColumn(1)]
    public bool Label { get; set; }
}

public class SentimentPrediction : SentimentData
{
    [ColumnName("PredictedLabel")]
    public bool PredictedLabel { get; set; }

    public float Score { get; set; }
    public float Probability { get; set; }
}
