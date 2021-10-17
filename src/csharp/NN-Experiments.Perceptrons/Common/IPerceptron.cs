using NNExperiments.Common.ActivationFunctions;
using NNExperiments.Perceptrons.Layers;
using System;
using System.Collections.Generic;

namespace NNExperiments.Perceptrons.Common
{
    public interface IPerceptron : IPerceptronBase
    {
        IList<ILayerBase> Layers { get; set; }

        IPerceptron Add(ILayerBase layer);

        IPerceptron Add(IEnumerable<ILayerBase> layers);

        IPerceptron AddLayer(double[][] layerWeights, double[] layerBias);

        IPerceptron AddLayer(double[][] layerWeights, double[] layerBias, ActivationFunction activationFunction);

        IPerceptron AddLayer(double[][] layerWeights, Random random);

        IPerceptron AddLayer(double[][] layerWeights, ActivationFunction activationFunction, Random random);

        IPerceptron AddLayer(double[,] layerWeights, double[] layerBias);

        IPerceptron AddLayer(double[,] layerWeights, double[] layerBias, ActivationFunction activationFunction);

        IPerceptron AddLayer(double[,] layerWeights, Random random);

        IPerceptron AddLayer(double[,] layerWeights, ActivationFunction activationFunction, Random random);

        IPerceptron AddLayer(int numberOfNeurons, int numberOfInputs, Random random);

        IPerceptron AddLayer(int numberOfNeurons, int numberOfInputs, ActivationFunction activationFunction, Random random);

        IPerceptron Build();
    }
}
