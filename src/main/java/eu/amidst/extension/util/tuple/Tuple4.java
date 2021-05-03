package eu.amidst.extension.util.tuple;

/**
 * Generic tuple implementation for 4 objects
 */
public class Tuple4<A, B, C, D> {

    private A first;

    private B second;

    private C third;

    private D fourth;


    public Tuple4(A first, B second, C third, D fourth) {
        super();
        this.first = first;
        this.second = second;
        this.third = third;
        this.fourth = fourth;
    }

    public A getFirst() {
        return first;
    }

    public B getSecond() {
        return second;
    }

    public C getThird() {
        return third;
    }

    public D getFourth() {
        return fourth;
    }

    public boolean equals(Object other) {

        if (other == null)
            return false;

        if (other instanceof Tuple4) {
            Tuple4 otherTuple4 = (Tuple4) other;

            if(this.first.equals(otherTuple4.first)
                    && this.second.equals(otherTuple4.second)
                    && this.third.equals(otherTuple4.third)
                    && this.fourth.equals(otherTuple4.fourth))
                return true;
        }

        return false;
    }

    public int hashCode() {
        int hashFirst = this.first != null ? this.first.hashCode() : 0;
        int hashSecond = this.second != null ? this.second.hashCode() : 0;
        int hashThird = this.third != null ? this.third.hashCode() : 0;
        int hashFourth = this.fourth != null ? this.fourth.hashCode() : 0;

        return (hashFirst + hashSecond + hashThird + hashFourth) * 7;
    }

    public String toString() {
        return "(" + this.first + ", " + this.second + ", " + this.third + "," + this.fourth + ")";
    }
}
