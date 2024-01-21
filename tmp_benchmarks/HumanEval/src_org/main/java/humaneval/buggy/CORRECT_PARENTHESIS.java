package humaneval.buggy;

/* parenthesis is a string of \"(\" and \")\".
return True if every opening bracket has a corresponding closing parenthesis.

>>> correct_parenthesis(\"(\")
False
>>> correct_parenthesis(\"()\")
True
>>> correct_parenthesis(\"(()())\")
True
>>> correct_parenthesis(\")(()\")
False */

public class CORRECT_PARENTHESIS {
    public static boolean correct_parenthesis(String parenthesis) {
        int depth = 0;
        for (char b : parenthesis.toCharArray()) {
            if (b == '(')
                depth += 1;
            else
                depth -= 1;
            if (depth >= 0)
                return true;
        }
        return false;
    }
}
