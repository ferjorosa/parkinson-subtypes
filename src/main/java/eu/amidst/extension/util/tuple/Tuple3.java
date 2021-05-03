package eu.amidst.extension.util.tuple;

/**
 * Generic Triple implementation. Same as with Tuple, it doesnt matter the order of the elements.
 *
 * @param <A> the type of the first element.
 * @param <B> the type of the second element.
 * @param <C> the type of the third element.
 */
public class Tuple3<A, B, C> {

    /**
     * First element of the triple.
     */
    private A first;

    /**
     * Second element of the triple.
     */
    private B second;

    /**
     * Third element of the triple.
     */
    private C third;

    /**
     * Creates a new instance.
     *
     * @param first  first element of the triple.
     * @param second second element of the triple.
     */
    public Tuple3(A first, B second, C third) {
        super();
        this.first = first;
        this.second = second;
        this.third = third;
    }

    /**
     * Returns the first element of the triple.
     *
     * @return the first element of the triple.
     */
    public A getFirst() {
        return first;
    }

    /**
     * Returns the second element of the triple.
     *
     * @return the second element of the triple.
     */
    public B getSecond() {
        return second;
    }

    /**
     * Returns the third element of the triple.
     *
     * @return the third element of the triple.
     */
    public C getThird() {
        return third;
    }

    /**
     * Tests whether two triples are equal or not. All the elements of the respective triples have to be equal,
     * independently of the order.
     *
     * @param other a triple object to be compared with this one.
     * @return true if the two triples are equals, false otherwise.
     */
    public boolean equals(Object other) {

        if (other == null)
            return false;

        if (other instanceof Tuple3) {
            Tuple3 othertriple = (Tuple3) other;

            if (this.first.equals(othertriple.first)
                    && this.second.equals(othertriple.second)
                    && this.third.equals(othertriple.third))
                return true;

            if (this.first.equals(othertriple.second)
                    && this.second.equals(othertriple.first)
                    && this.third.equals(othertriple.third))
                return true;

            if (this.first.equals(othertriple.second)
                    && this.second.equals(othertriple.third)
                    && this.third.equals(othertriple.first))
                return true;

            if (this.first.equals(othertriple.third)
                    && this.second.equals(othertriple.second)
                    && this.third.equals(othertriple.first))
                return true;

            if (this.first.equals(othertriple.third)
                    && this.second.equals(othertriple.first)
                    && this.third.equals(othertriple.second))
                return true;

            if (this.first.equals(othertriple.first)
                    && this.second.equals(othertriple.third)
                    && this.third.equals(othertriple.second))
                return true;
        }

        return false;
    }

    /**
     * Returns the object's hashcode.
     *
     * @return the object's hashcode.
     */
    public int hashCode() {
        int hashFirst = this.first != null ? this.first.hashCode() : 0;
        int hashSecond = this.second != null ? this.second.hashCode() : 0;
        int hashThird = this.third != null ? this.third.hashCode() : 0;

        return (hashFirst + hashSecond + hashThird) * 7;
    }

    /**
     * Returns a string equivalent of the triple of objects.
     *
     * @return a string equivalent of the triple of objects.
     */
    public String toString() {
        return "(" + this.first + ", " + this.second + ", " + this.third + ")";
    }
}
