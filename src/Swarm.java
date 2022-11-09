import java.util.ArrayList;
import java.util.Random;
public class Swarm {

    String[] mlp;
    String neural_type;
    double biases;
    double p1 = 1.7;
    double p2 = 2.4;
    double c1 = 2;
    double c2 = 1;

    double upper_limit_vector = 1.0;
    double kenhedy;
    double inertia_w;


    int num_particle;
    int maxEpoch;



    double gbest = Double.MAX_VALUE;
    Matrix[] weight_gbest;
//    Matrix[] vector_weight;
    Random r = new Random();
    individual[] seach_space;

    individual[] selection_pool ;

    ArrayList<Double[]> train_dataset;
    ArrayList<Double[]> train_desired_data;

    ArrayList<Double[]> train_result = new ArrayList<>();

    ArrayList< Pair<Double , individual>> scoreBoard = new ArrayList<>();

//    double prob_mul = 0.01;
//    double prob_parent = 0.3;
//    double best_mlp_score = 0.0;


    public Swarm(String _mlp , double _biases , int _population , int _maxGeneration){
        this.neural_type = _mlp;
        this.mlp = _mlp.split(",");;
        this.biases = _biases;
        this.num_particle = _population;
        this.maxEpoch = _maxGeneration;

        double p = p1+p2;
        double in =  Math.pow(p,2) - (4*p);
        double abs =   Math.abs(in);
        double root_2 = Math.sqrt(abs)/2;
        this.kenhedy = ( 1.0 - (1/p) +root_2);


        this.inertia_w = 0.5 * (c1+c2) - 1;
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
        this.seach_space = init;
        this.selection_pool = new individual[num_particle];


        this.weight_gbest = newWeight(seach_space[0]);
//        this.vector_weight =  newWeight(seach_space[0]);
    }

    public void run_gen(){
//        System.out.println("train");
        init_Particle();

        for (int epoch = 0; epoch < maxEpoch; epoch++){

            Double[] result =  paticle_eval(train_dataset,train_desired_data);
            System.out.println(result[0] + "\t" + result[1] + "\t" + result[2] );
            train_result.add(result);


            //step 4
            for (individual particle : seach_space) {
                Matrix[] particle_weight = particle.get_weight();

                Matrix[] vector_weight = newWeight(particle);
                for (int layer = 0 ; layer < particle.layer_weight.length ; layer++) {
                    Matrix p_w = Matrix.sub_matrix(particle.weight_pbest[layer],particle_weight[layer]);
                    Matrix learn_m_p = Matrix.mul_matrix(p_w, 1.496180);

                    Matrix g_w = Matrix.sub_matrix(weight_gbest[layer],particle_weight[layer]);
                    Matrix learn_m_g = Matrix.mul_matrix(g_w, 1.496180);



                    Matrix old_vector_w = Matrix.mul_matrix(particle.vector_weight[layer],0.729844);

                    //TODO
                    Matrix vector_w = Matrix.plus_matrix(old_vector_w,learn_m_g);
//                    Matrix vector_w = Matrix.plus_3matrix(old_vector_w,learn_m_p,learn_m_g);

                    //clerc & kenhedy //TODO don't need
                    Matrix ken = Matrix.mul_matrix(vector_w , 1);

                    vector_weight[layer] = ken;
//                    System.out.println("");
                }
                particle.set_vector_weight(vector_weight);
            }

            //step 5
            for (individual particle : seach_space) {

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
        System.out.println("");

    }

    public void test(ArrayList<Double[]> dataset, ArrayList<Double[]> desired_data){
//        System.out.println("test");
        individual best_solution =  scoreBoard.get(scoreBoard.size()-1).particle;
        best_solution.test(dataset,desired_data);
//        System.out.println("test");

    }



    public Double[] paticle_eval(ArrayList<Double[]> dataset , ArrayList<Double[]> desired_data){
        double sum_fit = 0;
        double avg_fit;
        double min_fit = Double.MAX_VALUE;

        //eval fitness
        for (individual particle : seach_space) {
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
                Pair<Double , individual> score_particle = new Pair<>(fitness, particle);
                scoreBoard.add(score_particle);
            }

        }
        avg_fit = sum_fit / num_particle;
        return new Double[]{sum_fit, avg_fit, min_fit};
    }

    public double scaling(double error){
        return 1/(error + 0.01)  ;
    }

    public Matrix[] newWeight(individual blueprint){
        Matrix[] offsprong_weight = new Matrix[blueprint.neural_type.length-1];
        for (int layer = 0; layer < offsprong_weight.length; layer++) {
            Matrix weight = new Matrix(blueprint.neural_type[layer+1],blueprint.neural_type[layer] ,false);
            offsprong_weight[layer] = weight;
        }
        return offsprong_weight;
    }


}
