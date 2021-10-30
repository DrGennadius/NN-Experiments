using NNExperiments.Common.ActivationFunctions;
using NNExperiments.Perceptrons.Layers;
using System;
using System.Collections.Generic;

namespace NNExperiments.Perceptrons.Common
{
    /// <summary>
    /// Interface of perceptron.
    /// </summary>
    public interface IPerceptron : IBasicPerceptron
    {
        /// <summary>
        /// Layers of the perceptron.
        /// </summary>
        IList<ILayerBase> Layers { get; set; }

        /// <summary>
        /// Add the layer.
        /// </summary>
        /// <param name="layer"></param>
        /// <returns></returns>
        IPerceptron Add(ILayerBase layer);

        /// <summary>
        /// Add the layers.
        /// </summary>
        /// <param name="layers"></param>
        /// <returns></returns>
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

        /// <summary>
        /// Convert to a new instance of <see cref="IBasicPerceptron"/>.
        /// </summary>
        /// <returns></returns>
        IBasicPerceptron GetBasicPerceptron();
    }
}
