package main

import (
	"fmt"

	"github.com/tddhit/golearn/base"
	"github.com/tddhit/golearn/evaluation"
	"github.com/tddhit/golearn/knn"
)

func main() {
	rawData, err := base.ParseCSVToInstances("../datasets/iris_headers.csv", true)
	if err != nil {
		panic(err)
	}

	//Initialises a new KNN classifier
	cls := knn.NewKnnClassifier("euclidean", "linear", 2)

	//Do a training-test split
	trainData, testData := base.InstancesTrainTestSplit(rawData, 0.9)
	cls.Fit(trainData)

	//Calculates the Euclidean distance and returns the most popular label
	predictions, probas, detail, err := cls.PredictProba(testData)
	if err != nil {
		panic(err)
	}
	fmt.Println(predictions)
	fmt.Println(probas)
	fmt.Println(detail)

	// Prints precision/recall metrics
	confusionMat, err := evaluation.GetConfusionMatrix(testData, predictions)
	if err != nil {
		panic(fmt.Sprintf("Unable to get confusion matrix: %s", err.Error()))
	}
	fmt.Println(evaluation.GetSummary(confusionMat))
}
