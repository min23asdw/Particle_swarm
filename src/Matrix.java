import java.util.Random;

public class Matrix {
    double[][] data;
    int rows,cols;

    /**
     * W ji weight form input neuron   i to j
     * @param rows j node
     * @param cols i node
     */

    public Matrix(int rows, int cols , boolean random){
        data = new double[rows][cols];
        this.rows=rows;
        this.cols=cols;
        Random generator = new Random();

        if(random){
            for(int j=0;j<rows;j++)
            {
                for(int i=0;i<cols;i++)
                {
                    double ran = 0;
                    while(ran == 0){
                        ran = generator.nextDouble(-1,1);
                        data[j][i]=ran;
                    }
                }
            }
        }
    }

    public static Matrix plus_matrix(Matrix a, Matrix b) {
        Matrix temp=new Matrix(a.rows,a.cols , false);
        for(int j=0;j<a.rows;j++)
        {
            for(int i=0;i<a.cols;i++)
            {
                temp.data[j][i]=a.data[j][i]+b.data[j][i];
            }
        }
        return temp;
    }

    public static Matrix plus_3matrix(Matrix a, Matrix b , Matrix c) {
        Matrix temp=new Matrix(a.rows,a.cols , false);
        for(int j=0;j<a.rows;j++)
        {
            for(int i=0;i<a.cols;i++)
            {
                temp.data[j][i]=a.data[j][i]+b.data[j][i]+c.data[j][i];
            }
        }
        return temp;
    }

    public static Matrix sub_matrix(Matrix a, Matrix b) {
        Matrix temp=new Matrix(a.rows,a.cols , false);
        for(int j=0;j<a.rows;j++)
        {
            for(int i=0;i<a.cols;i++)
            {
                temp.data[j][i]=a.data[j][i]-b.data[j][i];
            }
        }
        return temp;
    }

    public static Matrix mul_matrix(Matrix a, double mul) {
        Matrix temp=new Matrix(a.rows,a.cols , false);
        for(int j=0;j<a.rows;j++)
        {
            for(int i=0;i<a.cols;i++)
            {
                temp.data[j][i]=a.data[j][i]*mul;
            }
        }
        return temp;
    }
    public void add(int row, int col, double value) {
        this.data[row][col] += value;
    }
}
