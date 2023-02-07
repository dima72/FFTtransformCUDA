using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Reflection;
using System.IO;
using ManagedCuda;
using ManagedCuda.BasicTypes;
using ManagedCuda.VectorTypes;
using ManagedCuda.CudaFFT;

namespace simpleCUFFT
{
    class Program
    {
        static void Main(string[] args)
        {
            string  a_InputFn = Directory.GetCurrentDirectory() + @"\input.txt";
            string  a_OutputFn = Directory.GetCurrentDirectory() + @"\output.txt";
            Console.WriteLine("[simpleCUFFT] is starting...");

            var assembly = Assembly.GetExecutingAssembly();

            //CudaContext 
            CudaContext ctx = new CudaContext(0);
            // Set current to CPU thread, mandatory for a PrimaryContext
            ctx.SetCurrent();// 

            string[] a_lines = System.IO.File.ReadAllLines(a_InputFn);
            var SIGNAL_SIZE = a_lines.Count();

            // Allocate host memory for the signalcuFloatComplex for complex multiplaction in reference host code
            cuFloatComplex[] h_signal = new cuFloatComplex[SIGNAL_SIZE]; //we use ...

            foreach (string line in a_lines) //
            {
                string a_line = line;
                float a_X = (float)Convert.ToDouble(Fetch(ref a_line, " "));
                float a_Y = (float)Convert.ToDouble(Fetch(ref a_line, " "));
                int a_Ind = Convert.ToUInt16(Fetch(ref a_line, " "));

                h_signal[a_Ind].real = a_X;
                h_signal[a_Ind].imag = a_Y;

                System.Console.WriteLine("X: {0}  Y:{1}  I : {2}", a_X, a_Y, a_Ind);
               
            }


            // Allocate device memory for signal
            CudaDeviceVariable<cuFloatComplex> d_signal = new CudaDeviceVariable<cuFloatComplex>(SIGNAL_SIZE);

            Console.WriteLine("h_signal before cufftExecC2C");
            for (int i = 0; i < SIGNAL_SIZE; i++)
            {
                Console.WriteLine("{0} {1}", h_signal[i].real, h_signal[i].imag);
            }
            // Copy host memory to device
            d_signal.CopyToDevice(h_signal);
            
            // CUFFT plan simple API
            CudaFFTPlan1D plan = new CudaFFTPlan1D(SIGNAL_SIZE, cufftType.C2C, 1);


            // Transform signal and kernel
            Console.WriteLine("Transforming signal cufftExecC2C");
            plan.Exec(d_signal.DevicePointer, TransformDirection.Forward);

            Console.WriteLine("Array after cufftExecC2C TransformDirection.Forward ");
            for (int i = 0; i < SIGNAL_SIZE; i++)
                Console.WriteLine("{0} {1}", d_signal[i].real, d_signal[i].imag);

            // Transform signal back
            Console.WriteLine("Transforming signal back cufftExecC2C");
            plan.Exec(d_signal.DevicePointer, TransformDirection.Inverse);

            // Copy device memory to host
            cuFloatComplex[] h_out_signal = d_signal;
            Console.WriteLine("");
            Console.WriteLine("Inversed matrix");

            using (StreamWriter fs = new StreamWriter(a_OutputFn))
            {
                string a_line;
                for (int i = 0; i < SIGNAL_SIZE; i++)
                {
                    a_line = string.Format("{0} {1}", h_out_signal[i].real, h_out_signal[i].imag);
                    Console.WriteLine(a_line);
                    fs.WriteLine(a_line);
                }
            }



            //Destroy CUFFT context
            plan.Dispose();
            d_signal.Dispose();
            ctx.Dispose();
        }


     

        

        static string Fetch(ref string AInput, string ADelim)
        {
            string Result = "";
            AInput = AInput.Trim();
            int a_DelimStartPos = AInput.IndexOf(ADelim);
            if (a_DelimStartPos <= 0)
            {
                Result = AInput;
                AInput = "";
                return Result;
            }
            Result = AInput.Substring(0, a_DelimStartPos);
            int a_DelimEndPos = a_DelimStartPos + ADelim.Length;
            AInput = AInput.Substring(a_DelimEndPos, AInput.Length - a_DelimEndPos);
            return Result;
        }
    }
}
