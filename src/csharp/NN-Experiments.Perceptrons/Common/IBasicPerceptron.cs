using NNExperiments.Common.Training;
using System;

namespace NNExperiments.Perceptrons.Common
{
    /// <summary>
    /// Interface of basic perceptron.
    /// </summary>
    public interface IBasicPerceptron : ICloneable
    {
        double[][][] Weights { get; }
        double[][] Biases { get; }
        PerceptronTopology Topology { get; set; }
        double MomentumRate { get; set; }

        /// <summary>
        /// Forward propagation.
        /// </summary>
        /// <param name="input">Input data.</param>
        /// <returns></returns>
        double[] Forward(double[] input);

        /// <summary>
        /// Backward propagation.
        /// </summary>
        /// <param name="targetOutput">Target output data.</param>
        /// <param name="alpha">Alpha.</param>
        void Backward(double[] targetOutput, double alpha);

        /// <summary>
        /// Transfer base data from the other perceptron.
        /// </summary>
        /// <param name="otherPerceptron"></param>
        void TransferFrom(IBasicPerceptron otherPerceptron);

        /// <summary>
        /// Transfer base data to the other perceptron.
        /// </summary>
        /// <param name="otherPerceptron"></param>
        void TransferTo(IBasicPerceptron otherPerceptron);
    }
}
