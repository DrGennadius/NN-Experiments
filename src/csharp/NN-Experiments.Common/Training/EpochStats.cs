namespace NNExperiments.Common.Training
{
    /// <summary>
    /// The epoch stats.
    /// </summary>
    public struct EpochStats
    {
        public int Epoch;

        public double Error;

        public EpochStats(int epoch, double error)
        {
            Epoch = epoch;
            Error = error;
        }

        public override string ToString()
        {
            return $"Epoch: {Epoch}; error: {Error};";
        }
    }
}
