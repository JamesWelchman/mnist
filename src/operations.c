#include <stdlib.h>

/* 3 x 3 convolution */
#define TOP_LEFT9 0
#define TOP_MIDDLE9 1
#define TOP_RIGHT9 2
#define MIDDLE_LEFT9 3
#define MIDDLE_MIDDLE9 4
#define MIDDLE_RIGHT9 5
#define BOTTOM_LEFT9 6
#define BOTTOM_MIDDLE9 7
#define BOTTOM_RIGHT9 8



void
convolute28(double *in, double* out, double* conv) {
	double *row1, *row2, *row3;
	int i, j;

	/* pointers to the start of rows */
	row1 = in;
	row2 = in + 28;
	row3 = row2 + 28;

	/* iterate over all rows */
	for (i = 0; i < 26; i++) {
		/* iterate over the column */
		for (j = 0; j < 26; j++) {
			out[(i * 26) + j] = (
				row1[j] * conv[TOP_LEFT9] +
				row1[j + 1] * conv[TOP_MIDDLE9] +
				row1[j + 2] * conv[TOP_RIGHT9] +
				row2[j] * conv[MIDDLE_LEFT9] +
				row2[j + 1] * conv[MIDDLE_MIDDLE9] +
				row2[j + 2] * conv[MIDDLE_RIGHT9] +
				row3[j] * conv[BOTTOM_LEFT9] +
				row3[j + 1] * conv[BOTTOM_MIDDLE9] +
				row3[j + 2] * conv[BOTTOM_RIGHT9]
			);
		}
		/* advance the row iters */
		row1 += 28;
		row2 += 28;
		row3 += 28;
	}
}

void
convoluted28_backprop(double *in, double* out, double* d_conv)
{
	int i, j, k, l, conv_ind, in_ind; 

	for (k = 0; k < 3; k++) {
		for (l = 0; l < 3; l++) {
			conv_ind = (k * 3) + 1;
			/* we're computing d_conv[(k * 3) + l] */
			d_conv[conv_ind] = 0;

			for (i = 0; i < 26; i++) {
				for (j = 0; j < 26; j++) {
					in_ind = ((i + k) * 28) + ((j + l) * 28);
					d_conv[conv_ind] += out[(i * 26) + j] * in[in_ind];
				}
			}
		}
	}
}

double
max(double a, double b, double c, double d)
{
	double m;

	m = a;

	if (m < b) {
		m = b;
	}

	if (m < c) {
		m = c;
	}

	if (m < d) {
		m = d;
	}

	return m;
}

void
pool26(double* in, double* out)
{
	double *row1, *row2;
	int i, j;

	row1 = in;
	row2 = in + 26;

	for (i = 0; i < 13; i++) {

		/* iterate over the columns */
		for (j = 0; j < 13; j++) {
			out[(i * 13) + j] = max(
				row1[j * 2],
				row1[(j * 2) + 1],
				row2[j * 2],
				row2[(j * 2) + 1]
			);
		}

		/* advance the row iters */
		row1 = row2 + 26; 
		row2 = row1 + 26;
	}
}

int
pool26_backprop(double *in, double* out, double* back, double* d_loss)
{
	int i, j;
	double val, loss, *row1, *row2, *back1, *back2;
	int found;

	row1 = in;
	row2 = row1 + 26;
	back1 = back;
	back2 = back1 + 26;

	for (i = 0; i < 26 * 26; i++) {
		back[i] = 0;
	}

	/* iterate over the rows */
	for (i = 0; i < 13; i++) {
		/* iterate over the columns */
		for (j = 0; j < 13; j++) {
			val = out[(i * 13) + j];
			found = 0;
			loss = d_loss[(i * 13) + j];

			/* One or more of the four input pixels *
			 * must be equal to val */
			if (row1[j * 2] == val) {
				/* set the index of back to the value of d_loss */
				back1[j * 2] = loss;
				found = 1;
			}

			if (row1[(j * 2) + 1] == val) {
				/* set the index of back to the value of d_loss */
				back1[(j * 2) + 1] = loss;
				found = 1;
			}

			if (row2[j * 2] == val) {
				/* set the index of back to the value of d_loss */
				back2[j * 2] = loss;
				found = 1;
			}

			if (row2[(j * 2) + 1] == val) {
				/* set the index of back to the value of d_loss */
				back2[(j * 2) + 1] = loss;
				found = 1;
			}

			if (found == 0) {
				/* we want to panic here */
				return -1;
			}
		}

		/* Advance the row iters */
		row1 = row2 + 26;
		row2 = row1 + 26;
		back1 = back2 + 26;
		back2 = back1 + 26;
	}

	return 0;
}

void
convolute13(double *in, double* out, double* conv) {
	double *row1, *row2, *row3;
	int i, j;

	/* pointers to the start of rows */
	row1 = in;
	row2 = in + 13;
	row3 = row2 + 13;

	/* iterate over all rows */
	for (i = 0; i < 11; i++) {
		/* iterate over the column */
		for (j = 0; j < 11; j++) {
			out[(i * 11) + j] = (
				row1[j] * conv[TOP_LEFT9] +
				row1[j + 1] * conv[TOP_MIDDLE9] +
				row1[j + 2] * conv[TOP_RIGHT9] +
				row2[j] * conv[MIDDLE_LEFT9] +
				row2[j + 1] * conv[MIDDLE_MIDDLE9] +
				row2[j + 2] * conv[MIDDLE_RIGHT9] +
				row3[j] * conv[BOTTOM_LEFT9] +
				row3[j + 1] * conv[BOTTOM_MIDDLE9] +
				row3[j + 2] * conv[BOTTOM_RIGHT9]
			);
		}
		/* advance the row iters */
		row1 += 13;
		row2 += 13;
		row3 += 13;
	}
}

void
convoluted13_backprop(double *in, double* out, double* d_conv)
{
	int i, j, k, l, conv_ind, in_ind; 

	for (k = 0; k < 3; k++) {
		for (l = 0; l < 3; l++) {
			conv_ind = (k * 3) + 1;
			/* we're computing d_conv[(k * 3) + l] */
			d_conv[conv_ind] = 0;

			for (i = 0; i < 11; i++) {
				for (j = 0; j < 11; j++) {
					in_ind = ((i + k) * 13) + ((j + l) * 13);
					d_conv[conv_ind] += out[(i * 11) + j] * in[in_ind];
				}
			}
		}
	}
}

/* we need this buffer */
double CONV13_BACK_BUF[15 * 15];

void convoluted13_backprop_input(double* conv,
	                             double *loss,
	                             double* out)
{
	int i, j;
	double *row1, *row2, *row3;

	/* zero out + buffer */
	for (i = 0; i < 13 * 13; i++)
		out[i] = 0;

	for (i = 0; i < 15 * 15; i++)
		CONV13_BACK_BUF[i] = 0;

	/* populate the CONV13 buf */
	for (i = 0; i < 11; i++) {
		for (j = 0; j < 11; j++) {
			CONV13_BACK_BUF[((i + 2) * 15) + (j + 2)] = (
				loss[(i * 11) + j]
			);
		}
	}

	row1 = &CONV13_BACK_BUF[0];
	row2 = row1 + 15;
	row3 = row2 + 15;

	/* Apply the 9x9 conv matrix to the 15x15 block */
	for (i = 0; i < 13; i++) {
		for (j = 0; j < 13; j++) {

			out[(i * 13) + j] = (
				row1[j] * conv[BOTTOM_RIGHT9] +
				row1[j + 1] * conv[BOTTOM_MIDDLE9] +
				row1[j + 2] * conv[BOTTOM_LEFT9] +
				row2[j] * conv[MIDDLE_RIGHT9] +
				row2[j + 1] * conv[MIDDLE_MIDDLE9] +
				row2[j + 2] * conv[MIDDLE_LEFT9] +
				row3[j] * conv[TOP_MIDDLE9] +
				row3[j + 1] * conv[TOP_MIDDLE9] +
				row3[j + 2] * conv[TOP_LEFT9]
			);
		}
		row1 += 15;
		row2 += 15;
		row3 += 15;
	}
}

void
pool11(double* in, double* out) {
	double *row1, *row2;
	int i, j;
	double m;

	row1 = in;
	row2 = in + 11;

	for (i = 0; i < 5; i++) {

		/* iterate over the columns */
		for (j = 0; j < 5; j++) {
			m = max(
				row1[j * 2],
				row1[(j * 2) + 1],
				row2[j * 2],
				row2[(j * 2) + 1]
			);
			out[(i * 5) + j] = m;
		}

		/* advance the row iters */
		row1 = row2 + 11; 
		row2 = row1 + 11;
	}
}

int
pool11_backprop(double *in, double* out, double* back, double* d_loss)
{
	int i, j;
	double val, loss, *row1, *row2, *back1, *back2;
	int found;

	row1 = in;
	row2 = row1 + 11;
	back1 = back;
	back2 = back1 + 11;

	for (i = 0; i < 121; i++) {
		back[i] = 0;
	}

	/* iterate over the rows */
	for (i = 0; i < 5; i++) {
		/* iterate over the columns */
		for (j = 0; j < 5; j++) {
			val = out[(i * 5) + j];
			found = 0;
			loss = d_loss[(i * 5) + j];

			/* One or more of the four input pixels *
			 * must be equal to val */
			if (row1[j * 2] == val) {
				/* set the index of back to the value of d_loss */
				back1[j * 2] = loss;
				found = 1;
			}

			if (row1[(j * 2) + 1] == val) {
				/* set the index of back to the value of d_loss */
				back1[(j * 2) + 1] = loss;
				found = 1;
			}

			if (row2[j * 2] == val) {
				/* set the index of back to the value of d_loss */
				back2[j * 2] = loss;
				found = 1;
			}
			if (row2[(j * 2) + 1] == val) {
				/* set the index of back to the value of d_loss */
				back2[(j * 2) + 1] = loss;
				found = 1;
			}

			if (found == 0) {
				/* we want to panic here */
				return -1;
			}
		}

		/* Advance the row iters */
		row1 = row2 + 11;
		row2 = row1 + 11;
		back1 = back2 + 11;
		back2 = back1 + 11;
	}

	return 0;
}
