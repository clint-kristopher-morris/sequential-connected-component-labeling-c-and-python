#include <iostream>
using namespace std;
const int w = 5, h = 5;
int input[w][h] =  {{1,0,0,0,1},
					{1,1,0,1,1},
					{0,1,0,0,1},
					{1,1,1,0,1},
					{0,0,0,1,0}};
int component[w*h];
void doUnion(int a, int b)
{

  // get the root component of a and b, and set the one's parent to the other

  for (int x = 0; x < w; x++)
    {
      for (int y = 0; y < h; y++)
      {
         int c = x*h + y;
         cout << component[c];
         cout << ' ';
         cout << ' ';
         cout << ' ';
         cout << ' ';
         cout << ' ';
         cout << ' ';
         }
      cout << '\n';
    }
  cout << '\n';
  cout << '\n';
  cout << '\n';
  while (component[a] != a)
    a = component[a];
  while (component[b] != b)
    b = component[b];
  component[b] = a;
}

void unionCoords(int x, int y, int x2, int y2)
{
  if (y2 < h && x2 < w && input[x][y] && input[x2][y2])
    doUnion(x*h + y, x2*h + y2);
}

int main()
{
  // set up input

  for (int i = 0; i < w*h; i++)
    component[i] = i;

  for (int x = 0; x < w; x++)
  for (int y = 0; y < h; y++)
  {
    unionCoords(x, y, x+1, y);
    unionCoords(x, y, x, y+1);
  }

	for (int x = 0; x < w; x++)
	{
	  for (int y = 0; y < h; y++)
	  {
	  	 if (input[x][y] == 0)
	  	 {
	  	    cout << (char)('#');
	  	    cout << "\n";
	  	    continue;
	  	 }
	     int c = x*h + y; // counts left to right top to bottom [0,1,2,3,4]
	     //cout << (int) (c);
	     cout << (string) ("BEFORE: ");
         cout << component[c];
	     while (component[c] != c)
            {
                cout << (string) ("------------------------------\n");
                c = component[c];
            }
         cout << (string) ("AFTER: ");
         cout << component[c];
	  }
	  cout << "\n";
	}
}
