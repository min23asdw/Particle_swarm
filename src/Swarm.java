import java.util.ArrayList;
import java.util.Random;
public class Swarm {

    String[] mlp;
    String neural_type;
    double biases;
    double p1 = 1.496180;
    double p2 = 1.496180;

    double inertia_w = 0.729844;

    int num_particle;
    int maxEpoch;

    double gbest = Double.MAX_VALUE;
    Matrix[] weight_gbest;
    Random r = new Random();
    individual[] particle_pool;

    ArrayList<Double[]> train_dataset;
    ArrayList<Double[]> train_desired_data;

    ArrayList<Double[]> train_result = new ArrayList<>();

    ArrayList< Pair<Double , individual>> scoreBoard = new ArrayList<>();

    public Swarm(String _mlp , double _biases , int _population , int _maxGeneration){
        this.neural_type = _mlp;
        this.mlp = _mlp.split(",");;
        this.biases = _biases;
        this.num_particle = _population;
        this.maxEpoch = _maxGeneration;
    }

    public void settraindata(ArrayList<Double[]> _train_dataset, ArrayList<Double[]> _train_desired_data) {
        this.train_dataset = _train_dataset;
        this.train_desired_data = _train_desired_data;
    }

    public void init_Particle(){
        individual[] init =  new individual[num_particle];
        for(int indi = 0; indi < num_particle; indi++){
            individual indiler = new individual( neural_type ,biases );
            init[indi] = indiler;
        }
        this.particle_pool = init;

        this.weight_gbest = newWeight(particle_pool[0]);
    }

    public void particle_search(){

        init_Particle();

        for (int epoch = 0; epoch < maxEpoch; epoch++){

            Double[] result =  particle_eval(train_dataset,train_desired_data);
            if(epoch==maxEpoch-1){
//                System.out.print(result[0] + "\t" + result[1] + "\t" + result[2] + "\t" );
                System.out.print(result[2] + "\t" );
            }
            train_result.add(result);

            //step 4
            updateVector();

            //step 5
            updatePosition();
        }
    }

    public void test(ArrayList<Double[]> dataset, ArrayList<Double[]> desired_data){
        individual best_solution =  scoreBoard.get(scoreBoard.size()-1).particle;
        best_solution.test(dataset,desired_data);

    }

    public void updateVector(){
        for (individual particle : particle_pool) {
            Matrix[] particle_weight = particle.get_weight();

            Matrix[] vector_weight = newWeight(particle);
            for (int layer = 0 ; layer < particle.layer_weight.length ; layer++) {
                Matrix p_w = Matrix.sub_matrix(particle.weight_pbest[layer],particle_weight[layer]);
                Matrix learn_m_p = Matrix.mul_matrix(p_w, p1);

                Matrix g_w = Matrix.sub_matrix(weight_gbest[layer],particle_weight[layer]);
                Matrix learn_m_g = Matrix.mul_matrix(g_w, p2);

                Matrix old_vector_w = Matrix.mul_matrix(particle.vector_weight[layer],inertia_w);

                Matrix vector_w = Matrix.plus_3matrix(old_vector_w,learn_m_p,learn_m_g);

                vector_weight[layer] = vector_w;
            }
            particle.set_vector_weight(vector_weight);
        }
    }

    public void updatePosition(){
        for (individual particle : particle_pool) {

            Matrix[] particle_weight = particle.get_weight();
            Matrix[] vector_weight = particle.get_vector_weight();

            Matrix[] new_particle_weight = newWeight(particle);
            for (int layer = 0 ; layer < particle.layer_weight.length ; layer++) {
                Matrix a = Matrix.plus_matrix(particle_weight[layer], vector_weight[layer]);
                new_particle_weight[layer] = a;
            }
            particle.set_weight(new_particle_weight);
        }
    }

    public Double[] particle_eval(ArrayList<Double[]> dataset , ArrayList<Double[]> desired_data){
        double sum_fit = 0;
        double avg_fit;
        double min_fit = Double.MAX_VALUE;

        //eval fitness
        for (individual particle : particle_pool) {
            double error_mlp =   particle.eval(dataset , desired_data);

            double fitness = error_mlp;
            if(fitness < min_fit ){
                min_fit = fitness;
            }

            sum_fit += fitness;
            //step 3.1
            if(fitness < particle.pbest){
                particle.set_pbest(fitness);
                particle.set_weight_pbest(particle.get_weight());
            }
            //step 3.2
            if(fitness < gbest){
                gbest = fitness;
                weight_gbest = particle.get_weight();
                Pair<Double , individual> good_particle = new Pair<>(fitness, particle);
                scoreBoard.add(good_particle);
            }

        }
        avg_fit = sum_fit / num_particle;
        return new Double[]{sum_fit, avg_fit, min_fit};
    }

    public Matrix[] newWeight(individual blueprint){
        Matrix[] new_weight = new Matrix[blueprint.neural_type.length-1];
        for (int layer = 0; layer < new_weight.length; layer++) {
            Matrix weight = new Matrix(blueprint.neural_type[layer+1],blueprint.neural_type[layer] ,false);
            new_weight[layer] = weight;
        }
        return new_weight;
    }


}
