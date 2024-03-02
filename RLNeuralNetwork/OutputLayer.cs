using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLNeuralNetwork
{
    class OutputLayer : NetworkLayer
    {
        public OutputLayer(int inputs, int nodes, double baseLearningRate, double baseWeightMulti) : base(inputs, nodes, baseLearningRate, baseWeightMulti) { }
        public OutputLayer(double[,] weights, double baseLearningRate) : base(weights, baseLearningRate) { }

        internal override double activationFunction(double x)
        {
            return x;
        }

        internal override double activationDerivative(double x)
        {
            return 1;
        }
    }
}
