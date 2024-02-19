using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace RLNeuralNetwork
{
    internal class WideDeepNeuralNetwork : NeuralNetwork
    {
        public WideDeepNeuralNetwork(HiddenLayer[] hiddenLayers, WideDeepOutputLayer outputLayer) : base(hiddenLayers, outputLayer) { }

        public double[] wideDeepFeedForward(double[] inputs, double[] wideInputs)
        {
            if (outputLayer is WideDeepOutputLayer)
            {
                for (int i = 0; i < hiddenLayers.Length; i++)
                {
                    inputs = hiddenLayers[i].feedforward(inputs);
                }
                return ((WideDeepOutputLayer)outputLayer).feedforward(inputs, wideInputs);
            }
            else
            {
                throw new Exception("Output layer is not a wide-deep output layer");
            }
        }

        public double[,] optimisedFeedForward(double[] equalInputs, double[,] uniqueInputs, double[] wideEqualInputs, double[,] wideUniqueInputs)
        {
            //NOTE: assumes final layer uses linear/identity activation function.
            double[,] predicts = optimisedFeedForward(equalInputs, uniqueInputs);
            int baseK = wideEqualInputs.Length;
            for (int i = 0; i < predicts.GetLength(0); i++)
            {
                for (int j = 0; j < predicts.GetLength(1); j++)
                {
                    for (int k = 0; k < wideEqualInputs.Length; k++)
                    {
                        predicts[i, j] += wideEqualInputs[k] * ((WideDeepOutputLayer)outputLayer).wideWeights[k, j];
                    }
                    for (int k = 0; k < wideUniqueInputs.GetLength(1); k++)
                    {
                        predicts[i, j] += wideUniqueInputs[i, k] * ((WideDeepOutputLayer)outputLayer).wideWeights[baseK + k, j];
                    }
                }
            }
            return predicts;
        }

        public override void backPropagate(double[] inputs, double[] actualValues, double learningRateMulti, bool printLoss = false)
        {

            base.backPropagate(inputs, actualValues, learningRateMulti, printLoss);
        }

        public void backPropagate(double[] inputs, double[] wideInputs, double[] actualValues, double learningRateMulti, bool printLoss = false)
        {
            double[] ff = wideDeepFeedForward(inputs, wideInputs);
            double[] losses = new double[ff.Length];
            for (int i = 0; i < losses.Length; i++)
            {
                losses[i] = ff[i] - actualValues[i];
            }
            if (printLoss)
            {
                Console.WriteLine(losses[0]);
            }
            outputLayer.backPropagate(losses, learningRateMulti);
        }
    }
}
