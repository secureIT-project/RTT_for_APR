package humaneval;

public class TEST_CORRECT_PARENTHESIS {
    @org.junit.Test(timeout = 3000)
    public void test_0() throws java.lang.Exception {
        boolean result = humaneval.buggy.CORRECT_PARENTHESIS.correct_parenthesis("()");
        org.junit.Assert.assertEquals(
            result, true
        );
    }

    @org.junit.Test(timeout = 3000)
    public void test_1() throws java.lang.Exception {
        boolean result = humaneval.buggy.CORRECT_PARENTHESIS.correct_parenthesis("(()())");
        org.junit.Assert.assertEquals(
            result, true
        );
    }

    @org.junit.Test(timeout = 3000)
    public void test_2() throws java.lang.Exception {
        boolean result = humaneval.buggy.CORRECT_PARENTHESIS.correct_parenthesis("()()(()())()");
        org.junit.Assert.assertEquals(
            result, true
        );
    }

    @org.junit.Test(timeout = 3000)
    public void test_3() throws java.lang.Exception {
        boolean result = humaneval.buggy.CORRECT_PARENTHESIS.correct_parenthesis("()()((()()())())(()()(()))");
        org.junit.Assert.assertEquals(
            result, true
        );
    }

    @org.junit.Test(timeout = 3000)
    public void test_4() throws java.lang.Exception {
        boolean result = humaneval.buggy.CORRECT_PARENTHESIS.correct_parenthesis("((()())))");
        org.junit.Assert.assertEquals(
            result, false
        );
    }

    @org.junit.Test(timeout = 3000)
    public void test_5() throws java.lang.Exception {
        boolean result = humaneval.buggy.CORRECT_PARENTHESIS.correct_parenthesis(")(()");
        org.junit.Assert.assertEquals(
            result, false
        );
    }

    @org.junit.Test(timeout = 3000)
    public void test_6() throws java.lang.Exception {
        boolean result = humaneval.buggy.CORRECT_PARENTHESIS.correct_parenthesis("(");
        org.junit.Assert.assertEquals(
            result, false
        );
    }

    @org.junit.Test(timeout = 3000)
    public void test_7() throws java.lang.Exception {
        boolean result = humaneval.buggy.CORRECT_PARENTHESIS.correct_parenthesis("((((");
        org.junit.Assert.assertEquals(
            result, false
        );
    }

    @org.junit.Test(timeout = 3000)
    public void test_8() throws java.lang.Exception {
        boolean result = humaneval.buggy.CORRECT_PARENTHESIS.correct_parenthesis(")");
        org.junit.Assert.assertEquals(
            result, false
        );
    }

    @org.junit.Test(timeout = 3000)
    public void test_9() throws java.lang.Exception {
        boolean result = humaneval.buggy.CORRECT_PARENTHESIS.correct_parenthesis("(()");
        org.junit.Assert.assertEquals(
            result, false
        );
    }

    @org.junit.Test(timeout = 3000)
    public void test_10() throws java.lang.Exception {
        boolean result = humaneval.buggy.CORRECT_PARENTHESIS.correct_parenthesis("()()(()())())(()");
        org.junit.Assert.assertEquals(
            result, false
        );
    }

    @org.junit.Test(timeout = 3000)
    public void test_11() throws java.lang.Exception {
        boolean result = humaneval.buggy.CORRECT_PARENTHESIS.correct_parenthesis("()()(()())()))()");
        org.junit.Assert.assertEquals(
            result, false
        );
    }
}
