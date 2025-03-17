/* A Bison parser, made by GNU Bison 3.8.2.  */

/* Bison interface for Yacc-like parsers in C

   Copyright (C) 1984, 1989-1990, 2000-2015, 2018-2021 Free Software Foundation,
   Inc.

   This program is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   This program is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with this program.  If not, see <https://www.gnu.org/licenses/>.  */

/* As a special exception, you may create a larger work that contains
   part or all of the Bison parser skeleton and distribute that work
   under terms of your choice, so long as that work isn't itself a
   parser generator using the skeleton or a modified version thereof
   as a parser skeleton.  Alternatively, if you modify or redistribute
   the parser skeleton itself, you may (at your option) remove this
   special exception, which will cause the skeleton and the resulting
   Bison output files to be licensed under the GNU General Public
   License without this special exception.

   This special exception was added by the Free Software Foundation in
   version 2.2 of Bison.  */

/* DO NOT RELY ON FEATURES THAT ARE NOT DOCUMENTED in the manual,
   especially those whose name start with YY_ or yy_.  They are
   private implementation details that can be changed or removed.  */

#ifndef YY_YY_Y_TAB_H_INCLUDED
# define YY_YY_Y_TAB_H_INCLUDED
/* Debug traces.  */
#ifndef YYDEBUG
# define YYDEBUG 0
#endif
#if YYDEBUG
extern int yydebug;
#endif

/* Token kinds.  */
#ifndef YYTOKENTYPE
# define YYTOKENTYPE
  enum yytokentype
  {
    YYEMPTY = -2,
    YYEOF = 0,                     /* "end of file"  */
    YYerror = 256,                 /* error  */
    YYUNDEF = 257,                 /* "invalid token"  */
    JACOBIAN = 258,                /* JACOBIAN  */
    DOUBLE = 259,                  /* DOUBLE  */
    FUNCTION = 260,                /* FUNCTION  */
    DEFVAR = 261,                  /* DEFVAR  */
    DEFRAD = 262,                  /* DEFRAD  */
    DEFFIX = 263,                  /* DEFFIX  */
    SETVAR = 264,                  /* SETVAR  */
    SETRAD = 265,                  /* SETRAD  */
    SETFIX = 266,                  /* SETFIX  */
    HESSIAN = 267,                 /* HESSIAN  */
    STOICMAT = 268,                /* STOICMAT  */
    STOCHASTIC = 269,              /* STOCHASTIC  */
    DECLARE = 270,                 /* DECLARE  */
    INITVALUES = 271,              /* INITVALUES  */
    EQUATIONS = 272,               /* EQUATIONS  */
    FAMILIES = 273,                /* FAMILIES  */
    LUMP = 274,                    /* LUMP  */
    INIEQUAL = 275,                /* INIEQUAL  */
    EQNEQUAL = 276,                /* EQNEQUAL  */
    EQNCOLON = 277,                /* EQNCOLON  */
    LMPCOLON = 278,                /* LMPCOLON  */
    LMPPLUS = 279,                 /* LMPPLUS  */
    SPCPLUS = 280,                 /* SPCPLUS  */
    SPCEQUAL = 281,                /* SPCEQUAL  */
    FAMCOLON = 282,                /* FAMCOLON  */
    ATOMDECL = 283,                /* ATOMDECL  */
    CHECK = 284,                   /* CHECK  */
    CHECKALL = 285,                /* CHECKALL  */
    REORDER = 286,                 /* REORDER  */
    MEX = 287,                     /* MEX  */
    DUMMYINDEX = 288,              /* DUMMYINDEX  */
    EQNTAGS = 289,                 /* EQNTAGS  */
    LOOKAT = 290,                  /* LOOKAT  */
    LOOKATALL = 291,               /* LOOKATALL  */
    TRANSPORT = 292,               /* TRANSPORT  */
    TRANSPORTALL = 293,            /* TRANSPORTALL  */
    MONITOR = 294,                 /* MONITOR  */
    USES = 295,                    /* USES  */
    SPARSEDATA = 296,              /* SPARSEDATA  */
    WRITE_ATM = 297,               /* WRITE_ATM  */
    WRITE_SPC = 298,               /* WRITE_SPC  */
    WRITE_MAT = 299,               /* WRITE_MAT  */
    WRITE_OPT = 300,               /* WRITE_OPT  */
    INITIALIZE = 301,              /* INITIALIZE  */
    XGRID = 302,                   /* XGRID  */
    YGRID = 303,                   /* YGRID  */
    ZGRID = 304,                   /* ZGRID  */
    USE = 305,                     /* USE  */
    LANGUAGE = 306,                /* LANGUAGE  */
    INTFILE = 307,                 /* INTFILE  */
    DRIVER = 308,                  /* DRIVER  */
    RUN = 309,                     /* RUN  */
    INLINE = 310,                  /* INLINE  */
    ENDINLINE = 311,               /* ENDINLINE  */
    PARAMETER = 312,               /* PARAMETER  */
    SPCSPC = 313,                  /* SPCSPC  */
    INISPC = 314,                  /* INISPC  */
    INIVALUE = 315,                /* INIVALUE  */
    EQNSPC = 316,                  /* EQNSPC  */
    EQNSIGN = 317,                 /* EQNSIGN  */
    EQNCOEF = 318,                 /* EQNCOEF  */
    RATE = 319,                    /* RATE  */
    LMPSPC = 320,                  /* LMPSPC  */
    SPCNR = 321,                   /* SPCNR  */
    ATOMID = 322,                  /* ATOMID  */
    LKTID = 323,                   /* LKTID  */
    MNIID = 324,                   /* MNIID  */
    INLCTX = 325,                  /* INLCTX  */
    INCODE = 326,                  /* INCODE  */
    SSPID = 327,                   /* SSPID  */
    EQNLESS = 328,                 /* EQNLESS  */
    EQNTAG = 329,                  /* EQNTAG  */
    EQNGREATER = 330,              /* EQNGREATER  */
    TPTID = 331,                   /* TPTID  */
    USEID = 332,                   /* USEID  */
    FLUX = 333,                    /* FLUX  */
    AUTOREDUCE = 334,              /* AUTOREDUCE  */
    PLSPC = 335,                   /* PLSPC  */
    UPPERCASEF90 = 336,            /* UPPERCASEF90  */
    MINVERSION = 337               /* MINVERSION  */
  };
  typedef enum yytokentype yytoken_kind_t;
#endif

/* Value type.  */
#if ! defined YYSTYPE && ! defined YYSTYPE_IS_DECLARED
union YYSTYPE
{
#line 76 "scan.y"

  char str[1000];

#line 150 "y.tab.h"

};
typedef union YYSTYPE YYSTYPE;
# define YYSTYPE_IS_TRIVIAL 1
# define YYSTYPE_IS_DECLARED 1
#endif


extern YYSTYPE yylval;


int yyparse (void);


#endif /* !YY_YY_Y_TAB_H_INCLUDED  */
