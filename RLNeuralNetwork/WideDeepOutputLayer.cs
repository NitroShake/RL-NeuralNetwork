using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLNeuralNetwork
{
    class WideDeepOutputLayer : OutputLayer
    {
        internal double[,] wideWeights;
        internal double[] lastWideInputs;
        public WideDeepOutputLayer(double[,] weights, double[,] wideWeights, double baseLearningRate) : base(weights, baseLearningRate)
        {
            this.wideWeights = wideWeights;
        }
        public WideDeepOutputLayer(int inputs, int wideInputs, int nodes, double baseLearningRate) : base(inputs, nodes, baseLearningRate)
        {
            wideWeights = new double[wideInputs, nodes];
            Random r = new Random();
            for (int i = 0; i < weights.GetLength(0); i++)
            {
                for (int j = 0; j < weights.GetLength(0); j++)
                {
                    wideWeights[i, j] = (double)r.NextDouble() * 0.00001;
                }
            }
        }

        bool feedForwardReady = false;
        public override double[] feedforward(double[] previousLayer)
        {
            if (!feedForwardReady)
            {
                throw new Exception("Use feedforward(previousLayer, wideLayer) instead.");
            }
            else
            {
                return base.feedforward(previousLayer);
            }
        }

        public double[] feedforward(double[] previousLayer, double[] wideLayer)
        {
            //NOTE: This implementation only works if the activation function is f(x) = x.
            feedForwardReady = true;
            double[] nodes = feedforward(previousLayer);
            feedForwardReady = false;
            for (int i = 0; i < nodes.Length; i++)
            {
                for (int j = 0; j < wideLayer.Length; j++)
                {
                    nodes[i] += wideLayer[j] * wideWeights[j, i];
                }
            }
            lastWideInputs = wideLayer;
            return nodes;
        }

        internal override void backPropagate(double[] losses, double learningRateMulti)
        {
            for (int i = 0; i < losses.Length; i++)
            {
                for (int j = 0; j < wideWeights.GetLength(0); j++)
                {
                    wideWeights[j, i] = wideWeights[j, i] - (lastWideInputs[j] * learningRateMulti * losses[i]);
                }
            }
            base.backPropagate(losses, learningRateMulti);
        }
    }
}
