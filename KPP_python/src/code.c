/******************************************************************************

  KPP - The Kinetic PreProcessor
        Builds simulation code for chemical kinetic systems

  Copyright (C) 1995-1996 Valeriu Damian and Adrian Sandu
  Copyright (C) 1997-2005 Adrian Sandu

  KPP is free software; you can redistribute it and/or modify it under the
  terms of the GNU General Public License as published by the Free Software
  Foundation (http://www.gnu.org/copyleft/gpl.html); either version 2 of the
  License, or (at your option) any later version.

  KPP is distributed in the hope that it will be useful, but WITHOUT ANY
  WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
  FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
  details.

  You should have received a copy of the GNU General Public License along
  with this program; if not, consult http://www.gnu.org/copyleft/gpl.html or
  write to the Free Software Foundation, Inc., 59 Temple Place - Suite 330,
  Boston, MA  02111-1307,  USA.

  Adrian Sandu
  Computer Science Department
  Virginia Polytechnic Institute and State University
  Blacksburg, VA 24060
  E-mail: sandu@cs.vt.edu

******************************************************************************/


#include "gdata.h"
#include "code.h"
#include <unistd.h>
#include <string.h>

/*            NONE, ADD, SUB, MUL, DIV, POW, CONST, ELM, VELM, MELM, EELM  */
int PRI[] = {   10,   1,   1,   2,   2,   3,    10,  10,   10,   10,   10 };

// Function prototpyes
void (*WriteElm)( NODE *n );
void (*WriteSymbol)( int op );
void (*WriteAssign)( char* lval, char* rval );
void (*WriteComment)( char *fmt, ...  );
void (*WriteOMPThreadPrivate)( char *fmt, ...  );
void (*Declare)( int v );
void (*ExternDeclare)( int v );
void (*GlobalDeclare)( int v );
void (*InitDeclare)( int var, int nv, void * values );
void (*DeclareConstant)( int v, char *val );
void (*FunctionStart)( int f, int *vars );
void (*FunctionPrototipe)( int f, ... );
void (*FunctionBegin)( int f, ... );
void (*FunctionEnd)( int f );

// Local definitions of function prototypes just for code.c
// This will avoid "implicit function declaration" warnings
//   -- Bob Yantosca (29 Apr 2022)
void FatalError( int status, char *fmt, ... );
void Message( char *fmt, ...  );

NODE * substList;
int substENABLED = 1;
int crtop = NONE;
char *outBuf;
char *outBuffer;

VARIABLE cnst = { "", CONST, REAL, 0, 0 };
VARIABLE expr = { "", EELM, 0, 0, 0 };
VARIABLE *varTable[ MAX_VAR ] = { &cnst, &expr };

int IsConst( NODE *n, float val );
NODE * BinaryOp( int op, NODE *n1, NODE *n2 );
int NodeCmp( NODE *n1, NODE *n2 );
NODE * NodeCopy( NODE *n1 );
void WriteNode( NODE *n );
void WriteOp( int op );
void ExpandElm( NODE * n );
int ExpandNode( NODE *n, int lastop );
NODE * LookUpSubst( NODE *n );

FILE * param_headerFile   = 0;
FILE * initFile = 0;
FILE * driverFile = 0;
FILE * integratorFile = 0;
FILE * linalgFile = 0;
FILE * functionFile = 0;
FILE * jacobianFile = 0;
FILE * rateFile = 0;
FILE * stoichiomFile = 0;
FILE * utilFile = 0;
FILE * sparse_dataFile = 0;
FILE * sparse_jacFile = 0;
FILE * sparse_hessFile = 0;
FILE * sparse_stoicmFile = 0;
FILE * stochasticFile = 0;
FILE * global_dataFile = 0;
FILE * hessianFile = 0;
FILE * logFile = 0;
FILE * makeFile = 0;
FILE * monitorFile = 0;
FILE * mex_funFile = 0;
FILE * mex_jacFile = 0;
FILE * mex_hessFile = 0;

FILE * currentFile;

int ident = 0;

FILE * UseFile( FILE * file )
{
FILE *oldf;
    if (file == NULL) {
      printf("\n\nKPP Warning (internal): trying to UseFile NULL file pointer!\n");
    }
    oldf = currentFile;
    currentFile = file;
    return oldf;
}


void OpenFile( FILE **fpp, char *name, char * ext, char * identity )
{
char bufname[MAX_PATH];
char buf[MAX_PATH];
time_t t;
int blength;

  time( &t );
  sprintf( bufname, "%s%s", name, ext );
  if( *fpp ) fclose( *fpp );
  *fpp = fopen( bufname, "w" );
  if ( *fpp == 0 )
    FatalError(3,"%s: Can't create file", bufname );

  UseFile( *fpp );

  WriteDelim();
  WriteComment("");
  WriteComment("%s",identity);
  WriteComment("");
  WriteComment("Generated by KPP-%s symbolic chemistry Kinetics PreProcessor",
                KPP_VERSION );
  WriteComment("      (https:/github.com/KineticPreProcessor/KPP");
  WriteComment("KPP is distributed under GPL, the general public licence");
  WriteComment("      (http://www.gnu.org/copyleft/gpl.html)");
  WriteComment("(C) 1995-1997, V. Damian & A. Sandu, CGRER, Univ. Iowa" );
  WriteComment("(C) 1997-2022, A. Sandu, Michigan Tech, Virginia Tech" );
  WriteComment("    With important contributions from:" );
  WriteComment("       M. Damian,   Villanova University, Philadelphia, PA, USA");
  WriteComment("       R. Sander,   Max-Planck Institute for Chemistry, Mainz, Germany");
  WriteComment("       M. Long,     Renaissance Fiber, LLC, North Carolina, USA");
  WriteComment("       H. Lin,      Harvard University, Cambridge, MA, USA");
  WriteComment("       R. Yantosca, Harvard University, Cambridge, MA, USA");
  WriteComment("");

  WriteComment("%-20s : %s", "File", bufname );
  strcpy( buf, (char*)ctime( &t ) );
  buf[ (int)strlen(buf) - 1 ] = 0;
//===========================================================================
// MODIFICATION by Bob Yantosca (11 Feb 2021)
//
// Do not write out changeable parameters such as file creation time
// and working directory.  These will cause Git to interpret changed
// files as new files that need to be committed.
//
//  WriteComment("%-20s : %s", "Time", buf );
//  WriteComment("%-20s : %s", "Working directory", getcwd(buf, MAX_PATH) );
//===========================================================================
  WriteComment("%-20s : %s", "Equation file", eqFileName );
  WriteComment("%-20s : %s", "Output root filename", rootFileName );
  WriteComment("");
  WriteDelim();
  NewLines(1);
/* Include Headers in .c  Files, except Makefile */
  blength = strlen(bufname);
  if ( (bufname[blength-2]=='.')&&(bufname[blength-1]=='c') ) {
    C_Inline("#include <stdio.h>");
    C_Inline("#include <stdlib.h>");
    C_Inline("#include <math.h>");
    C_Inline("#include <string.h>");
    C_Inline("#include \"%s_Parameters.h\"", rootFileName);
    C_Inline("#include \"%s_Global.h\"", rootFileName);
    if( useJacSparse )
       C_Inline("#include \"%s_Sparse.h\"", rootFileName);
    }
  NewLines(2);
}

void AllowBreak()
{
  *(outBuffer-1) |= 0x80;
}

void bprintf( char *fmt, ... )
{
Va_list args;

  if ( !fmt ) return;
  Va_start( args, fmt );
  vsprintf( outBuffer, fmt, args );
  va_end( args );
  outBuffer += strlen( outBuffer );
}

void FlushBuf()
{
char *p;

  p = outBuf;
  while( *p )
    *p++ &= ~0x80;
  fprintf( currentFile, "%s", outBuf );
  outBuffer = outBuf;
  *outBuffer = 0;
}

void FlushThisBuf( char * buf )
{
char *p;

  p = buf;
  while( *p )
    *p++ &= ~0x80;
  fprintf( currentFile, "%s", buf );
}

void WriteDelim()
{
  WriteComment("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~");
}

void NewLines( int n )
{
  for( ; n > 0; n-- )
    bprintf("\n");

  FlushBuf();
}

void IncludeFile( char * fname )
{
FILE *fp;
#define MAX_LINE 200
char line[ MAX_LINE ];


  fp = fopen( fname, "r" );
  if ( fp == 0 )
    FatalError(3,"%s: Can't read file", fname );

  FlushBuf();

  while( !feof(fp) ) {
    *line = '\0';
    fgets( line, MAX_LINE, fp );
    fputs( line, currentFile );
  }

  fclose( fp );
}

void IncludeCode( char* fmt, ... )
{

Va_list args;
char buf[MAX_PATH];
char cmd[500];
char tmp[500];
static char tmpfile[] = "kppfile.tmp";
FILE * fp;
int isMakefile;
char *FUNCALL[] = { "FUN_SPLIT( Y, FIX, RCONST, Ydot, P, D )", /* index 0 = split */ 
                    "FUN( Y, FIX, RCONST, Ydot )"};            /* index 1 = aggregate */

  Va_start( args, fmt );
  vsprintf( buf, fmt, args );
  va_end( args );

  //=========================================================================
  // MODIFICATION by Bob Yantosca (03 May 2022)
  //
  // Update the switch statement so that it looks for Makefiles with the
  // naming convention Makefile_f90, etc., but for other files with the
  // naming convention util.f90, etc.
  //
  // The only exception is when upperCaseF90 == 1, then we need to include
  // util/Makefile_upper_F90 instead of util/Makefile_f90.  Rename the
  // Makefile_F90 to Makefile_upper_F90 to avoid issues on case-insensitive
  // operating systems such as MacOS X.
  //
  // Also, We cannot use self-referential sprintf statements such as
  // "sprintf( buf, "%s", buf );", as this is considered undefined
  // behavior by the C language standard.  Replace instances of these
  // with strncat or strncpy statements.  (NOTE: For srncat/strncpy,
  // the number of characters must be one more than the text to allow
  // for the null string `\0', which terminates the string.)
  //=========================================================================
  isMakefile = ( strstr( buf, "Makefile" ) != NULL );  // Is it a makefile?

  switch( useLang ) {
    case F90_LANG:
      if ( upperCaseF90 )
	if ( isMakefile ) { strncat( buf, "_upper_F90", 11 ); break; }
        else              { strncat( buf, ".f90", 5 ); break; } // *.f90
      else
	if ( isMakefile ) { strncat( buf, "_f90", 5 ); break; } // Makefile_f90
        else              { strncat( buf, ".f90", 5 ); break; } // *.f90
    case F77_LANG:
      if ( isMakefile )   { strncat( buf, "_f",   3 ); break; } // Makefile_f
      else                { strncat( buf, ".f",   3 ); break; } // *.f
    case C_LANG:
      if ( isMakefile )   { strncat( buf, "_c",   3 ); break; } // Makefile_c
      else                { strncat( buf, ".c",   3 ); break; } // *.c
    case MATLAB_LANG:
      if ( isMakefile )   { strncat( buf, "_m",   3 ); break; } // Makefile_m
      else                { strncat( buf, ".m",   3 ); break; } // *.m
    default:
      printf("\n Language '%d' not implemented!\n",useLang); exit(1);
  }

  // Open Makefile
  fp = fopen( buf, "r" );
  if ( fp == 0 )
    FatalError(3,"%s: Can't read file", buf );
  fclose(fp);

  // Create sed command to replace KPP_ROOT w/ the model name
  strncpy( cmd, "sed ", 5 );

  sprintf( tmp, " -e 's/KPP_ROOT/%s/g'", rootFileName );
  strncat( cmd, tmp, strlen(tmp)+1 );

  sprintf( tmp, " -e 's/KPP_NVAR/%d/g'", VarNr );
  strncat( cmd, tmp, strlen(tmp)+1 );

  sprintf( tmp, " -e 's/KPP_NFIX/%d/g'", FixNr );
  strncat( cmd, tmp, strlen(tmp)+1 );

  sprintf( tmp, " -e 's/KPP_NSPEC/%d/g'", SpcNr );
  strncat( cmd, tmp, strlen(tmp)+1 );

  sprintf( tmp, " -e 's/KPP_NREACT/%d/g'", EqnNr );
  strncat( cmd, tmp, strlen(tmp)+1 );

  sprintf( tmp, " -e 's/KPP_NONZERO/%d/g'", Jac_NZ );
  strncat( cmd, tmp, strlen(tmp)+1 );

  sprintf( tmp, " -e 's/KPP_LU_NONZERO/%d/g'", LU_Jac_NZ );
  strncat( cmd, tmp, strlen(tmp)+1 );

  sprintf( tmp, " -e 's/KPP_NHESS/%d/g'", Hess_NZ );
  strncat( cmd, tmp, strlen(tmp)+1 );

  // either CALL FUN or CALL FUN_SPLIT, depending on useAggregate:
  sprintf( tmp, " -e 's/KPP_FUN_OR_FUN_SPLIT/%s/g'", FUNCALL[useAggregate] );
  strncat( cmd, tmp, strlen(tmp)+1 );

  // Also replace KPP_REAL with the selected precision
  switch( useLang ) {
    case F77_LANG:
      sprintf( tmp, " -e 's/KPP_REAL/%s/g'", F77_types[real] );
      strncat( cmd, tmp, strlen(tmp)+1 );
      break;
    case F90_LANG:
      sprintf( tmp, " -e 's/KPP_REAL/%s/g'", F90_types[real] );
      strncat( cmd, tmp, strlen(tmp)+1 );
      break;
    case C_LANG:
      sprintf( tmp, " -e 's/KPP_REAL/%s/g'", C_types[real] );
      strncat( cmd, tmp, strlen(tmp)+1 );
      break;
    case MATLAB_LANG:
      break;
    default:
      printf("\n Language '%d' not implemented!\n",useLang);
      exit(1);
  }

  // Write final sed command using safe string functions
  sprintf( tmp, "%s",    cmd                );
  strncat( tmp, " ",     2                  );
  strncat( tmp, buf,     strlen(buf)+1      );
  strncat( tmp, " > ",   4                  );
  strncat( tmp, tmpfile, strlen(tmpfile)+ 1 );
  strncpy( cmd, tmp,     strlen(tmp)+1      );

  // Run sed command
  system( cmd );
  IncludeFile( tmpfile );
  sprintf( cmd, "rm -f %s", tmpfile );
  system( cmd );
}

void LogFunctionComment( int f, int *vars )
{
FILE *oldf;

  oldf = UseFile( logFile );
  FunctionStart( f, vars );
  FlushBuf();
  UseFile( oldf );
}

int DefineVariable( char * name, int t, int bt, int maxi, int maxj, char * comment, int attr )
{
int i;
VARIABLE * var;

  for( i = 0; i < MAX_VAR; i++ )
    if( varTable[ i ] == 0 ) break;

  if( varTable[ i ] != 0 ) {
    printf("\nVariable Table overflow");
    return -1;
  }

  //=========================================================================
  // MODIFICATION by Bob Yantosca (22 Apr 2022)
  //
  // Add attr to the variable structure (and also in code.h), so that we
  // can flag variables that need the F90 POINTER or TARGET attributes.
  // Also add "attr" as an integer argument to this routine.
  //   -- Bob Yantosca (25 Apr 2022)
  //=========================================================================
  var = (VARIABLE*) malloc( sizeof( VARIABLE ) );
  var->name = name;
  var->type = t;
  var->baseType = bt;
  var->maxi = maxi;
  var->maxj = maxj;
  var->value = -1;
  var->attr = attr;
  var->comment = comment;

  varTable[ i ] = var;
  return i;
}

void FreeVariable( int n )
{
  if( varTable[ n ] ) {
    free( varTable[ n ] );
    varTable[ n ] = 0;
  }
}

NODE * Elm( int v, ... )
{
Va_list args;
NODE *n;
ELEMENT *elm;
VARIABLE *var;

  var = varTable[ v ];
  n   = (NODE*)    malloc( sizeof(NODE) );
  elm = (ELEMENT*) malloc( sizeof(ELEMENT) );
  n->left = 0;
  n->right = 0;
  n->sign = 1;
  n->type = var->type;
  n->elm = elm;
  elm->var = v;

  Va_start( args, v );
  switch( var->type ) {
    case CONST: switch( var->baseType ) {
                  case REAL: elm->val.cnst = (float)va_arg( args, double );
                             break;
                  case INT:  elm->val.cnst = (float)va_arg( args, int );
                }
                if( elm->val.cnst < 0 ) {
                  elm->val.cnst = -elm->val.cnst;
                  n->sign = -1;
                }
                break;
    case ELM:
                break;
    case VELM:  elm->val.idx.i = va_arg( args, int );
                break;
    case MELM:  elm->val.idx.i = va_arg( args, int );
                elm->val.idx.j = va_arg( args, int );
                break;
    case EELM:  elm->val.expr = va_arg( args, char* );
                break;
  }
  va_end( args );

  return n;
}

int IsConst( NODE *n, float val )
{
  return ( ( n ) &&
           ( n->type == CONST ) &&
           ( n->elm->val.cnst == val )
         );
}

NODE * BinaryOp( int op, NODE *n1, NODE *n2 )
{
NODE *n;

  n   = (NODE*)    malloc( sizeof(NODE) );
  n->left = n1;
  n->right = n2;
  n->type = op;
  n->sign = 1;
  n->elm = 0;
  return n;
}

NODE * Add( NODE *n1, NODE *n2 )
{
  if( n1 == 0 ) return n2;
  if( n2 == 0 ) return n1;

  if( IsConst( n1, 0 ) ) {
    FreeNode( n1 );
    return n2;
  }
  if( IsConst( n2, 0 ) ) {
    FreeNode( n2 );
    return n1;
  }
  return BinaryOp( ADD, n1, n2 );
}

NODE * Sub( NODE *n1, NODE *n2 )
{
  if( n1 == 0 ) return BinaryOp( SUB, 0, n2 );
  if( n2 == 0 ) return n1;

  if( IsConst( n1, 0 ) ) {
    FreeNode( n1 );
    return  BinaryOp( SUB, 0, n2 );
  }
  if( IsConst( n2, 0 ) ) {
    FreeNode( n2 );
    return n1;
  }
  return BinaryOp( SUB, n1, n2 );
}

NODE * Mul( NODE *n1, NODE *n2 )
{
  if( n1 == 0 ) return n2;
  if( n2 == 0 ) return n1;

  if( IsConst( n1, 1 ) ) {
    n2->sign *= n1->sign;
    FreeNode( n1 );
    return n2;
  }
  if( IsConst( n2, 1 ) ) {
    n2->sign *= n1->sign;
    FreeNode( n2 );
    return n1;
  }
  if( IsConst( n1, 0 ) ) {
    FreeNode( n2 );
    return n1;
  }
  if( IsConst( n2, 0 ) ) {
    FreeNode( n1 );
    return n2;
  }

  return BinaryOp( MUL, n1, n2 );
}

NODE * Div( NODE *n1, NODE *n2 )
{
  if( n1 == 0 ) return BinaryOp( DIV, Const(1), n2 );
  if( n2 == 0 ) return n1;

  if( IsConst( n2, 1 ) ) {
    n2->sign *= n1->sign;
    FreeNode( n2 );
    return n1;
  }

  return BinaryOp( DIV, n1, n2 );
}

NODE * Pow( NODE *n1, NODE *n2 )
{
  if( n1 == 0 ) return n2;
  if( n2 == 0 ) return n1;
  return BinaryOp( POW, n1, n2 );
}

void FreeNode( NODE * n )
{
  if( n == 0 ) return;
  FreeNode( n->left );
  FreeNode( n->right );
  if( n->elm ) free( n->elm );
  free( n );
}

int NodeCmp( NODE *n1, NODE *n2 )
{
ELEMENT *elm1;
ELEMENT *elm2;

  if( n1 == n2 ) return 1;
  if( n1 == 0 ) return 0;
  if( n2 == 0 ) return 0;

  if( (n1->type % SUBST) != (n2->type % SUBST) ) return 0;

  elm1 = n1->elm;
  elm2 = n2->elm;

  if( elm1 == elm2 ) return 1;
  if( elm1 == 0 ) return 0;
  if( elm2 == 0 ) return 0;

  if( elm1->var != elm2->var )return 0;
  switch( n1->type ) {
    case CONST: if( elm1->val.cnst != elm2->val.cnst ) return 0;
		break;
    case ELM:   break;
    case VELM:  if( elm1->val.idx.i != elm2->val.idx.i ) return 0;
		break;
    case MELM:  if( elm1->val.idx.i != elm2->val.idx.i ) return 0;
		if( elm1->val.idx.j != elm2->val.idx.j ) return 0;
		break;
    case EELM:  if( strcmp( elm1->val.expr, elm2->val.expr ) != 0 ) return 0;
		break;
  }

  return 1;
}

NODE * NodeCopy( NODE *n1 )
{
NODE *n;
ELEMENT *elm;

  n   = (NODE*)    malloc( sizeof(NODE) );
  elm = (ELEMENT*) malloc( sizeof(ELEMENT) );
  *n = *n1;
  n->elm = elm;
  *n->elm = *n1->elm;
  return n;
}

void WriteNode( NODE *n )
{
  crtop = NONE;
  ExpandNode( n, NONE );
}

void WriteOp( int op )
{
  WriteSymbol( op );
  crtop = NONE;
}

void ExpandElm( NODE * n )
{
NODE *cn;

  if( substENABLED == 0 ) {
    WriteElm( n );
    return;
  }
  cn = LookUpSubst( n );
  if( cn == 0 ) {
    WriteElm( n );
  } else {
    if( cn->type > SUBST ) {
      WriteElm( n );
    } else {
      cn->type += SUBST;
      WriteSymbol( O_PAREN );
      WriteNode( cn->right );
      WriteSymbol( C_PAREN );
      cn->type -= SUBST;
    }
  }
}

int ExpandNode( NODE *n, int lastop )
{
int needParen = 0;

  if( n == 0 ) return lastop;

  if( ( n->left ) &&
      ( PRI[ n->left->type ] < PRI[ n->type ] ) )
      needParen = 1;

  if( needParen ) {
    WriteOp( crtop );
    WriteSymbol( O_PAREN );
  }
  lastop = ExpandNode( n->left, lastop );
  if( needParen ) WriteSymbol( C_PAREN );

  switch( n->type ) {
    case ADD:
    case SUB:
    case MUL:
    case DIV:
    case POW:   crtop = n->type;
                break;
    case NONE:  printf("ERROR - null element");
    		break;
    case CONST:
    case ELM:
    case VELM:
    case MELM:
    case EELM:
		switch( crtop ) {
		  case MUL: case DIV: case POW:
		    WriteOp( crtop );
		    if ( n->sign == -1 ) {
		      WriteSymbol( O_PAREN );
		      WriteOp( SUB );
		      ExpandElm( n );
		      WriteSymbol( C_PAREN );
		    } else {
		      ExpandElm( n );
		    }
		    break;
		  case ADD:  if( n->sign == -1 )
			       crtop = SUB;
			     WriteOp( crtop );
			     ExpandElm( n );
			     break;
		  case SUB:  if( n->sign == -1 )
			       crtop = ADD;
			     WriteOp( crtop );
			     ExpandElm( n );
			     break;
		  case NONE: if( n->sign == -1 )
			       WriteOp( SUB );
			     ExpandElm( n );
			     break;
		}
		break;
  }

  if( ( n->right ) &&
      ( PRI[ n->right->type ] <= PRI[ n->type ] ) )
      needParen = 1;

  if( needParen ) {
    WriteOp( crtop );
    WriteSymbol( O_PAREN );
  }
  lastop = ExpandNode( n->right, n->type );
  if( needParen ) WriteSymbol( C_PAREN );
  return lastop;
}

void Assign( NODE *lval, NODE *rval )
{
char *ls;
char *rs;
char *olds;

  ls = (char*)malloc( MAX_OUTBUF );
  rs = (char*)malloc( MAX_OUTBUF );

  olds = outBuffer;
  outBuffer = ls;
  WriteNode( lval );
  outBuffer = rs;
  WriteNode( rval );
  outBuffer = olds;

  WriteAssign( ls, rs );

  free( rs );
  free( ls );
  FreeNode( lval );
  FreeNode( rval );
}

NODE * LookUpSubst( NODE *n )
{
NODE *cn;

  cn = substList;
  while( cn != 0 ) {
    if( NodeCmp( n, cn ) )
      return cn;
    cn = cn->left;
  }
  return 0;
}

void MkSubst( NODE *n1, NODE *n2 )
{
NODE *n;

  n = LookUpSubst( n1 );
  if( n == 0 ) {
    n = n1;
    n->left = substList;
    substList = n;
  } else {
    FreeNode( n->right );
    FreeNode( n1 );
  }
  n->right = n2;
}

void RmSubst( NODE *n )
{
NODE *pn;
NODE *cn;

  pn = 0;
  cn = substList;
  while( cn != 0 ) {
    if( NodeCmp( n, cn ) )
      break;
    pn = cn;
    cn = cn->left;
  }
  if( cn == 0 ) return;

  FreeNode( cn->right );
  if( pn )
    pn->left = cn->left;
  else
    substList = cn->left;

  cn->right = 0;
  cn->left = 0;
  FreeNode( cn );
}

void DisplaySubst()
{
NODE *n;

  n = substList;
  substENABLED = 0;
  while( n != 0 ) {
    printf("Subst: ");
    WriteElm( n );
    printf( " --> " );
    WriteNode( n->right );
    printf("\n");
    n = n->left;
  }
  substENABLED = 1;
}

void CommentFncBegin( int f, int *vars )
{
VARIABLE *var;
int narg;
int i;

  narg = varTable[ f ]->maxi;
  var = varTable[ f ];

  WriteDelim();
  WriteComment("");
  WriteComment("%s - %s", var->name, var->comment );
  WriteComment("  Arguments :");
  for( i = 0; i < narg; i++ ) {
    var = varTable[vars[i]];
    WriteComment("     %-10s- %s", var->name, var->comment );
  }
  WriteComment("");
  WriteDelim();
  NewLines(1);
}

void CommentFunctionBegin( int f, ... )
{
Va_list args;
int i;
int vars[20];
int narg;

  narg = varTable[ f ]->maxi;

  Va_start( args, f );
  for( i = 0; i < narg; i++ )
    vars[i] = va_arg( args, int );
  va_end( args );

  CommentFncBegin( f, vars );
}

void CommentFunctionEnd( int f )
{
  WriteComment("End of %s function", varTable[ f ]->name );
  WriteDelim();
  NewLines(2);
}
