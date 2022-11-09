// Pair class
class Pair<U, V>
{
    public final U score;       // the first field of a pair
    public final V particle;      // the second field of a pair

    // Constructs a new pair with specified values
    Pair(U fitscore, V particle)
    {
        this.score = fitscore;
        this.particle = particle;
    }

    @Override
    // Checks specified object is "equal to" the current object or not
    public boolean equals(Object o)
    {
        if (this == o) {
            return true;
        }

        if (o == null || getClass() != o.getClass()) {
            return false;
        }

        Pair<?, ?> pair = (Pair<?, ?>) o;

        // call `equals()` method of the underlying objects
        if (!score.equals(pair.score)) {
            return false;
        }
        return particle.equals(pair.particle);
    }

    @Override
    // Computes hash code for an object to support hash tables
    public int hashCode()
    {
        // use hash codes of the underlying objects
        return 31 * score.hashCode() + particle.hashCode();
    }

    @Override
    public String toString() {
        return "(" + score + ", " + particle + ")";
    }

    // Factory method for creating a typed Pair immutable instance
    public static <U, V> Pair <U, V> of(U a, V b)
    {
        // calls private constructor
        return new Pair<>(a, b);
    }
}