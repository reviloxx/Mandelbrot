using MPI;

using (new MPI.Environment(ref args))
{
    Intracommunicator comm = Communicator.world;
    int rank = comm.Rank;
    int size = comm.Size;

    int width = 800;
    int height = 600;
    int maxIter = 1000;
    double xmin = -2.5, xmax = 1.5, ymin = -2.0, ymax = 2.0;
    double xstep = (xmax - xmin) / width;
    double ystep = (ymax - ymin) / height;

    int rowsPerProcess = height / size;
    int[] localImage = new int[width * rowsPerProcess];

    int startRow = rank * rowsPerProcess;
    int endRow = (rank == size - 1) ? height : startRow + rowsPerProcess;

    for (int y = startRow; y < endRow; y++)
    {
        for (int x = 0; x < width; x++)
        {
            double a = xmin + x * xstep;
            double b = ymin + y * ystep;
            Complex c = new Complex(a, b);
            Complex z = new Complex(0, 0);
            int iter = 0;

            while (iter < maxIter && z.Magnitude < 2.0)
            {
                z = z * z + c;
                iter++;
            }

            localImage[(y - startRow) * width + x] = iter;
        }
    }

    int[] finalImage = null;
    if (rank == 0)
    {
        finalImage = new int[width * height];
    }

    comm.Gather(localImage, 0, localImage.Length, finalImage, 0, width * rowsPerProcess, 0);

    if (rank == 0)
    {
        // Handle remaining rows if the division is not perfect
        if (height % size != 0)
        {
            int remainingRows = height % size;
            int[] tempBuffer = new int[width * remainingRows];
            comm.Recv(tempBuffer, size - 1, 0);
            Array.Copy(tempBuffer, 0, finalImage, width * rowsPerProcess * size, tempBuffer.Length);
        }

        // Save or display the final image
        // (e.g., write it to a file or display it using a GUI library)
    }
    else if (rank == size - 1 && height % size != 0)
    {
        int remainingRows = height % size;
        int[] remainingImage = new int[width * remainingRows];
        Array.Copy(localImage, width * rowsPerProcess, remainingImage, 0, remainingImage.Length);
        comm.Send(remainingImage, 0, 0);
    }
}


struct Complex
{
    public double Real { get; }
    public double Imag { get; }

    public Complex(double real, double imag)
    {
        Real = real;
        Imag = imag;
    }

    public double Magnitude => Math.Sqrt(Real * Real + Imag * Imag);

    public static Complex operator +(Complex a, Complex b) => new Complex(a.Real + b.Real, a.Imag + b.Imag);
    public static Complex operator *(Complex a, Complex b) => new Complex(a.Real * b.Real - a.Imag * b.Imag, a.Real * b.Imag + a.Imag * b.Real);
}