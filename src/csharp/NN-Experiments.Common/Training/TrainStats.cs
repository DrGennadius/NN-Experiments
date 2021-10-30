using System;
using System.Collections.Generic;

namespace NNExperiments.Common.Training
{
    /// <summary>
    /// Train stats.
    /// </summary>
    public class TrainStats
    {
        public double LastError { get; set; } = double.MinValue;

        public int NumberOfEpoch { get; set; } = -1;

        public List<EpochStats> ErrorHistory { get; set; } = new();

        /// <summary>
        /// Train time.
        /// </summary>
        public TimeSpan Time { get; set; }

        public override string ToString()
        {
            return $"Last error: {LastError}; epoch: {NumberOfEpoch}; time: {Time}";
        }
    }
}
