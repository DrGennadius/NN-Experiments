namespace NNExperiments.Common.ActivationFunctions
{
    /// <summary>
    /// Type of activation function.
    /// </summary>
    public enum ActivationFunctionType
    {
        /// <summary>
        /// Sigmoid.
        /// </summary>
        Sigmoid,

        /// <summary>
        /// Hyperbolic Tangent.
        /// </summary>
        Tanh,

        /// <summary>
        /// Rectified Linear Unit.
        /// </summary>
        ReLU,
        /// <summary>
        /// Leaky Rectified Linear Unit.
        /// </summary>
        LReLU,

        /// <summary>
        /// Leaky Rectified Linear Unit with random.
        /// </summary>
        RandomLReLU,

        /// <summary>
        /// Custom function.
        /// </summary>
        Custom
    }
}
