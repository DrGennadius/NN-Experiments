using NNExperiments.Common.Training;
using System;

namespace NNExperiments.Perceptrons.Common
{
    public interface IPerceptronBase : ICloneable
    {
        double[][][] Weights { get; set; }
        double[][] Biases { get; set; }
        PerceptronTopology Topology { get; set; }
        double MomentumRate { get; set; }

        double[] Forward(double[] input);

        void Backward(double[] input, double[] targetOutput, double alpha);

        /// <summary>
        /// Training Perceptron.
        /// </summary>
        /// <param name="inputs">Inputs</param>
        /// <param name="outputs">Outputs</param>
        /// <param name="alpha">Learning rate.</param>
        /// <param name="targetError">Target error value.</param>
        /// <param name="maxEpoch">Max number epochs.</param>
        /// <param name="printError">Print error each epoch.</param>
        /// <returns></returns>
        TrainStats Train(double[,] inputs, double[,] outputs, double alpha, double targetError, int maxEpoch, bool printError = false);

        /// <summary>
        /// Training Perceptron.
        /// </summary>
        /// <param name="trainData">Data for training.</param>
        /// <param name="alpha">Learning rate.</param>
        /// <param name="targetError">Target error value.</param>
        /// <param name="maxEpoch">Max number epochs.</param>
        /// <param name="printError">Print error each epoch.</param>
        /// <returns></returns>
        TrainStats Train(TrainData trainData, double alpha, double targetError, int maxEpoch, bool printError = false);

        /// <summary>
        /// Transfer base data from the other perceptron.
        /// </summary>
        /// <param name="otherPerceptron"></param>
        void TransferFrom(IPerceptronBase otherPerceptron);

        /// <summary>
        /// Transfer base data to the other perceptron.
        /// </summary>
        /// <param name="otherPerceptron"></param>
        void TransferTo(IPerceptronBase otherPerceptron);
    }
}
